import json
import os
from typing import Any

import cv2
import depth_pro
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from kornia.filters import box_blur, canny
from kornia.morphology import closing, dilation, erosion, opening
from pycocotools.coco import COCO
from torch import Tensor
from tqdm import tqdm
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

device = "cuda:1"

# DepthPro
depthpro_cfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
depthpro_cfg.checkpoint_uri = "../../../github/ml-depth-pro/checkpoints/depth_pro.pt"
depthpro, depthpro_transform = depth_pro.create_model_and_transforms(depthpro_cfg)
depthpro.eval().to(device)

# ViTMatte
vitmatte_processor = VitMatteImageProcessor.from_pretrained(
    "hustvl/vitmatte-small-distinctions-646"
)
vitmatte = (
    VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-distinctions-646")
    .eval()
    .to(device)
)

# Hyper parameters
JSON_SAVE_FREQ = 1000
VERBOSE = False


def dprint(*arg):
    if VERBOSE:
        print(*arg, flush=True)
    return


def is_object_unoccluded(
    mask: Float[Tensor, "h w"],
    depth: Float[Tensor, "h w"],
    edge_iou_thresh: float = 0.05,
    depth_border_ratio_thresh: float = 0.99,
    depth_border_positive_thresh: float = 0.9,
) -> bool:
    assert mask.shape == depth.shape
    height, width = mask.shape
    mask = mask.reshape(1, 1, height, width)
    depth = depth.reshape(1, 1, height, width)

    # sanity check (ensure depth texture is similar to mask texture: this fails in case of printings or paintings)
    obj_depth_min = max(0, torch.quantile(depth[mask > 0.5], 0.05) - 1)
    obj_depth_max = torch.quantile(depth[mask > 0.5], 0.95) + 1
    dprint(f"[is_object_unoccluded] {obj_depth_min=}, {obj_depth_max=}")
    depth_normalized = torch.clip((depth - obj_depth_min) / (obj_depth_max - obj_depth_min), 0, 1)
    _, depth_edge = canny(depth_normalized)
    _, mask_edge = canny(mask)

    mask_depth_edge_iou = torch.sum(depth_edge * mask_edge) / (1e-6 + torch.sum(mask_edge))
    dprint(f"[is_object_unoccluded] {mask_depth_edge_iou.item()=}")
    if mask_depth_edge_iou < edge_iou_thresh:
        dprint(
            "[is_object_unoccluded] Mask and Depth doesn't align well. Occlusion cannot be judged from them."
        )
        return False

    # whole/partial check
    # extract mask border pixels
    mask_inner = mask > 0.5
    mask_outer = ~mask_inner
    mask_eroded = erosion(mask_inner.float(), torch.ones((3, 3), device=device))
    mask_border = mask_inner * (mask_eroded < 0.5)

    # depth comparison around mask borders
    depth_outer = torch.where(mask_inner, 0, depth)
    depth_outer_box_mean = box_blur(depth_outer, 3)
    mask_outer_box_mean = box_blur(mask_outer.float(), 3)
    depth_outer_mean = depth_outer_box_mean / (mask_outer_box_mean + 1e-6)

    # sum up the number of foreground border pixels
    is_foreground_flags = (depth_outer_mean >= depth_border_ratio_thresh * depth)[mask_border]
    is_foreground_mean = is_foreground_flags.float().mean()
    dprint(f"[is_object_unoccluded] {is_foreground_mean.item()=}")
    return is_foreground_mean > depth_border_positive_thresh


def is_simply_connected(mask: Float[np.ndarray, "h w"]):
    assert isinstance(mask, np.ndarray)
    totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(
        mask, 4, cv2.CV_32S
    )
    dprint(f"[is_simply_connected] {totalLabels=}")
    return totalLabels == 2


def process(img_id: int, img_dir: str, coco: COCO) -> list[dict[str, Any]]:
    L.seed_everything(0, verbose=False)  # just in case

    # load an image and annotations
    img_path = os.path.join(img_dir, coco.imgs[img_id]["file_name"])
    img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = torch.tensor(img_np, dtype=torch.float32, device=device).permute(2, 0, 1) / 255
    height, width, _ = img_np.shape

    # load annotations
    coco_annot_list = [ann for ann in coco.imgToAnns[img_id] if ann["iscrowd"] == 0]
    coco_annot_list_with_mask_type = []

    with torch.inference_mode():
        # DepthPro
        prediction = depthpro.infer(2 * img.unsqueeze(0) - 1, f_px=None)
        depth = prediction["depth"]  # Depth in [m].

        # refine and reorder masks from large to small (to prevent occlusion)
        for ann in coco_annot_list:
            msk_np = coco.annToMask(ann)

            # refine a mask
            if msk_np.sum() < 1024:  # COCO small masks are better not to be refined
                mask = torch.tensor(msk_np, dtype=torch.float32, device=device)
            else:
                msk_trimap_np = np.where(
                    msk_np > 0.5,
                    255,
                    np.where(cv2.dilate(msk_np, np.ones((5, 5))) > 0.5, 128, 0),
                )
                pixel_values = vitmatte_processor(
                    images=img_np, trimaps=msk_trimap_np, return_tensors="pt"
                ).pixel_values
                matting_outputs = vitmatte(pixel_values.to(device))
                mask = matting_outputs.alphas[..., :height, :width].reshape(height, width)

            # revert if something went wrong
            if torch.all(mask < 0.5):
                mask = torch.tensor(msk_np, dtype=torch.float32, device=device)
                if torch.all(mask < 0.5):  # It does exist
                    dprint(f"[process] {img_id=}, {ann['id']=}: Mask is all zeros.")
                    continue

            # check occlusion
            unoccluded = is_object_unoccluded(mask, depth) and is_simply_connected(
                coco.annToMask(ann)
            )
            ann["mask_type"] = "real_whole" if unoccluded else "real_partial"
            coco_annot_list_with_mask_type.append(ann)

    return coco_annot_list_with_mask_type


def save_json(
    img_list: list[int],
    annot_list: list[dict[str, Any]],
    json_outpath: str,
    coco: COCO,
):
    assert json_outpath.endswith(".json")

    # save into a json file
    json_data = {
        "images": img_list,
        "annotations": annot_list,
        "categories": list(coco.cats.values()),
    }
    with open(json_outpath, "w") as f:
        json.dump(json_data, f)


def load_json(json_path: str):
    with open(json_path) as f:
        json_data = json.load(f)

    img_list = json_data["images"]
    annot_list = json_data["annotations"]
    return img_list, annot_list


if __name__ == "__main__":
    coco_root_dir = "/media/ryotaro/ssd1/coco/"
    is_train_or_val = "train"

    img_dir = os.path.join(coco_root_dir, f"{is_train_or_val}2017")
    coco = COCO(os.path.join(coco_root_dir, f"annotations/instances_{is_train_or_val}2017.json"))
    annot_id_list = coco.getAnnIds(catIds=[])

    new_img_list = []
    new_img_set = set()
    new_annot_list = []

    # load json if exists (assuming the process was interrupted in the middle)
    json_outpath = os.path.join(
        coco_root_dir,
        f"annotations/instances_{is_train_or_val}2017_with_mask_type.json",
    )
    if os.path.isfile(json_outpath):
        dprint(
            f"[main] Loading {json_outpath}, assuming that the process was interrupted in the middle."
        )
        new_img_list, new_annot_list = load_json(json_outpath)
        new_img_set = {int(img_meta["id"]) for img_meta in new_img_list}

    for img_id in tqdm(coco.getImgIds()):
        if img_id in new_img_set:
            continue

        try:
            new_annot_list_per_image = process(img_id, img_dir, coco)
        except Exception as e:
            print(f"{img_id=}: {e}")
            raise e
        new_annot_list.extend(new_annot_list_per_image)

        # set complete
        new_img_list.append(coco.imgs[img_id])
        new_img_set.add(img_id)

    if new_img_list and len(new_img_list) % JSON_SAVE_FREQ == 0:
        save_json(new_img_list, new_annot_list, json_outpath, coco)

    # save into a json file
    save_json(new_img_list, new_annot_list, json_outpath, coco)
    print(f"Finished! {json_outpath=}")
