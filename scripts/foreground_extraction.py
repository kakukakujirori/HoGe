import glob
import json
import os
import sys

import cv2
import depth_pro
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools.coco import COCO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

sys.path.append("../")
from third_party.MoGe.moge.model import MoGeModel

device = "cuda:0"

checkpoint = "../../../github/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../../../github/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

depthpro_cfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
depthpro_cfg.checkpoint_uri = "../../../github/ml-depth-pro/checkpoints/depth_pro.pt"
depthpro, depthpro_transform = depth_pro.create_model_and_transforms(depthpro_cfg)
depthpro.eval().to(device)

moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").eval().to(device)

# coco = COCO("/media/ryotaro/ssd1/coco/annotations/instances_val2017.json")
# annot_id_list = coco.getAnnIds(catIds=[1])


def mask_edge_depth_edge_iou(mask: np.ndarray, depth: np.ndarray, verbose: bool = False) -> float:
    if np.all(mask <= 0.5):
        return 0

    test_size = 256
    object_depth_min = max(0, np.percentile(depth[mask > 0.5], 5) - 1)
    object_depth_max = np.percentile(depth[mask > 0.5], 95) + 1
    if verbose:
        print(f"{object_depth_min=}, {object_depth_max=}")
    depth_normalized = np.clip(
        (depth - object_depth_min) / (object_depth_max - object_depth_min), 0, 1
    )
    depth_normalized_uint8 = np.uint8(255 * cv2.resize(depth_normalized, (test_size, test_size)))
    depth_edge = cv2.Canny(depth_normalized_uint8, 100, 200) > 0

    mask_uint8 = np.uint8(255 * cv2.resize(mask, (test_size, test_size)))
    mask_edge = cv2.Canny(mask_uint8, 100, 200) > 0

    mask_depth_edge_iou = np.sum(depth_edge * mask_edge) / (1e-6 + np.sum(mask_edge))
    return mask_depth_edge_iou


def is_occluded(
    img: np.ndarray,
    coco_mask: np.ndarray,
    coco_sam_mask_iou_threshold: float = 0.9,
    mask_depth_edge_iou_threshold: float = 0.05,
    depth_delta: float = 0.99,
    foreground_ratio: float = 0.995,
    verbose: bool = False,
):
    # offset = 4
    # if coco_mask[:offset, :].max() > 0.5 or coco_mask[:, :offset].max() > 0.5 or coco_mask[-offset:, :].max() > 0.5 or coco_mask[:, -offset:].max() > 0.5:
    #     print(f"Mask touches the image border, skipping ({img_path})")
    #     return False, None, None, None

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # pick random points for SAM2 prompt (coco mask only was not performant)
        coco_mask_dist = cv2.distanceTransform(coco_mask, cv2.DIST_L2, 5)
        if np.all(coco_mask_dist <= 0):
            if verbose:
                print("Mask too small")
            return False, None, None, None
        coco_mask_indices = np.column_stack(
            np.where(coco_mask_dist > coco_mask_dist[coco_mask_dist > 0].mean())
        )

        sample_num = 16
        sample_indices = np.random.choice(
            len(coco_mask_indices), size=min(sample_num, len(coco_mask_indices)), replace=False
        )
        sampled_points = coco_mask_indices[sample_indices, ::-1]  # ij -> xy

        # segmentation
        sam2_predictor.set_image(img)
        lowres_mask = cv2.resize(coco_mask, (256, 256))
        lowres_mask = np.where(lowres_mask > 0.5, 32.0, -32.0)
        lowres_mask = cv2.GaussianBlur(lowres_mask, (3, 3), 0)
        masks, _, _ = sam2_predictor.predict(
            point_coords=sampled_points,
            point_labels=np.array([1] * len(sampled_points)),
            mask_input=lowres_mask[None, :, :],
            multimask_output=False,
            return_logits=False,
        )
        mask = masks[0]

        # sanity check (coco mask and sam mask should be similar)
        coco_sam_iou = (coco_mask * mask).sum() / np.maximum(coco_mask, mask).sum()
        if verbose:
            print(f"{coco_sam_iou=}, {coco_sam_mask_iou_threshold=}")
        if coco_sam_iou < coco_sam_mask_iou_threshold or np.isnan(coco_sam_iou).any():
            if verbose:
                print(f"Irregular mask refinement: {coco_sam_iou=}")
            return False, mask, None, None

        # depth estimation
        if True:
            # DepthPro
            image = depthpro_transform(img)
            prediction = depthpro.infer(image.to(device), f_px=None)
            depth = prediction["depth"].cpu().numpy()  # Depth in [m].
        else:
            # MoGe
            img_t = torch.tensor(img / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
            output = moge.infer(img_t)
            depth = (
                torch.where(
                    output["mask"], output["depth"], output["depth"].nan_to_num(0, 0, 0).max()
                )
                .cpu()
                .numpy()
            )

        # sanity check (ensure depth texture is similar to mask texture: this fails in case of printings or paintings)
        mask_depth_edge_iou = mask_edge_depth_edge_iou(mask, depth, verbose=verbose)
        if verbose:
            print(f"{mask_depth_edge_iou=}, {mask_depth_edge_iou_threshold=}")
        if mask_depth_edge_iou < mask_depth_edge_iou_threshold:
            if verbose:
                print(
                    f"Mask and Depth doesn't align well ({mask_depth_edge_iou=}). Occlusion cannot be judged from them."
                )
            return False, mask, depth, None

    # extract mask border pixels
    ksize = (3, 3)
    mask_binary = np.where(mask > 0.5, 1, 0).astype(np.uint8)
    mask_binary_eroded = cv2.erode(mask_binary, np.ones((3, 3)))
    mask_border = (mask_binary == 1) * (mask_binary_eroded == 0)

    # depth comparison around mask borders
    depth_outer = np.where(mask > 0.5, 0, depth)
    mask_outer = 1 - mask_binary
    depth_outer_box_mean = cv2.boxFilter(depth_outer, -1, ksize, normalize=False)
    mask_outer_box_mean = cv2.boxFilter(mask_outer, -1, ksize, normalize=False)
    depth_outer_mean = depth_outer_box_mean / (mask_outer_box_mean + 1e-6)

    # sum up the number of foreground border pixels
    is_foreground_flags = (depth_outer_mean >= depth_delta * depth)[mask_border]
    is_foreground_decision = is_foreground_flags.mean() > foreground_ratio
    if verbose:
        print(f"{is_foreground_flags.mean()=}, {is_foreground_decision=}")

    if not verbose:
        pass

    return (
        is_foreground_decision,
        mask,
        depth,
        (depth_outer_mean >= depth_delta * depth) * mask_border,
    )


COCO_SAM_MASK_IOU_THRESHOLD = 0.5
MASK_DEPTH_EDGE_IOU_THRESHOLD = 0.25
DEPTH_DELTA = 0.999  # 0.99
FOREGROUND_RATIO = 0.9  # 0.925


# rewrite json files
annot_file_path = "/media/ryotaro/ssd1/coco/annotations/instances_train2017.json"
coco = COCO(annot_file_path)
annot_id_list = coco.getAnnIds()


for i, annot_id in enumerate(tqdm(annot_id_list)):
    try:
        # load an image
        ann = coco.anns[annot_id]
        img_id = ann["image_id"]
        img_path = os.path.join(
            "/media/ryotaro/ssd1/coco/train2017", coco.imgs[img_id]["file_name"]
        )
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        coco_mask = coco.annToMask(ann)
        judgment, mask, depth, border_mask = is_occluded(
            img,
            coco_mask,
            coco_sam_mask_iou_threshold=COCO_SAM_MASK_IOU_THRESHOLD,
            mask_depth_edge_iou_threshold=MASK_DEPTH_EDGE_IOU_THRESHOLD,
            depth_delta=DEPTH_DELTA,
            foreground_ratio=FOREGROUND_RATIO,
            verbose=False,
        )

        ann["occluded"] = not judgment
        coco.anns[annot_id] = ann
    except RuntimeError as e:
        print(f"Error at {annot_id}: {e}")
        raise e


output_path = os.path.join(
    os.path.dirname(annot_file_path),
    os.path.basename(annot_file_path).replace(".json", "_with_occlusion_labels.json"),
)
with open(output_path, "w") as f:
    # ret = {"images": coco.imgs, "annotations": coco.anns, "categories": coco.cats}
    ret = {
        "images": list(coco.imgs.values()),
        "annotations": list(coco.anns.values()),
        "categories": list(coco.cats.values()),
    }
    json.dump(ret, f)
