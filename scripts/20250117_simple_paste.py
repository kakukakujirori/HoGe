import json
import os
import random
from collections import defaultdict
from typing import Any

import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pycocotools
import torch
import torch.nn.functional as F
import torchvision
from jaxtyping import Float, Int64
from kornia.color import hsv_to_rgb, rgb_to_hsv
from pycocotools.coco import COCO
from skimage.measure import find_contours
from torch import Tensor
from tqdm import tqdm
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

device = "cuda:0"

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
TARGET_SCALE_REL_RANGE = (1 / 2.5, 1 / 0.75)
LABEL_OFFSET = 1000

JSON_SAVE_FREQ = 1000

VERBOSE = False


def dprint(*arg):
    if VERBOSE:
        print(*arg, flush=True)
    return


def change_object_color_hsv(
    image: Float[torch.Tensor, "b h w 3"],
    mask: Float[torch.Tensor, "b h w"],
    sv_max_shift: float = 0.2,
) -> Float[torch.Tensor, "b h w 3"]:
    batch, height, width, channel = image.shape
    assert channel == 3
    assert mask.shape == (batch, height, width)
    assert 0 <= sv_max_shift <= 1
    mask = mask.reshape(batch, height, width, 1)

    hsv = rgb_to_hsv(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    sv_offset = (torch.rand(batch, 1, 1, 2, device=hsv.device) * 2 - 1) * sv_max_shift

    hsv_trans = hsv.clone()
    hsv_trans[..., 1:] = hsv[:, :, :, 1:] + sv_offset

    hsv = (1 - mask) * hsv + mask * hsv_trans
    return hsv_to_rgb(hsv.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


COCO2017_CATEGORIES = {
    1,  # person           (person)
    2,  # bicycle          (vehicle)
    3,  # car              (vehicle)
    4,  # motorcycle       (vehicle)
    # 5,  # airplane         (vehicle)
    6,  # bus              (vehicle)
    # 7,  # train            (vehicle)
    8,  # truck            (vehicle)
    # 9,  # boat             (vehicle)
    # 10,  # traffic light    (outdoor)
    # 11,  # fire hydrant     (outdoor)
    # 13,  # stop sign        (outdoor)
    # 14,  # parking meter    (outdoor)
    # 15,  # bench            (outdoor)
    16,  # bird             (animal)
    17,  # cat              (animal)
    18,  # dog              (animal)
    19,  # horse            (animal)
    20,  # sheep            (animal)
    21,  # cow              (animal)
    22,  # elephant         (animal)
    23,  # bear             (animal)
    24,  # zebra            (animal)
    25,  # giraffe          (animal)
    # 27,  # backpack         (accessory)
    # 28,  # umbrella         (accessory)
    # 31,  # handbag          (accessory)
    # 32,  # tie              (accessory)
    # 33,  # suitcase         (accessory)
    # 34,  # frisbee          (sports)
    # 35,  # skis             (sports)
    # 36,  # snowboard        (sports)
    # 37,  # sports ball      (sports)
    # 38,  # kite             (sports)
    # 39,  # baseball bat     (sports)
    # 40,  # baseball glove   (sports)
    # 41,  # skateboard       (sports)
    # 42,  # surfboard        (sports)
    # 43,  # tennis racket    (sports)
    # 44,  # bottle           (kitchen)
    # 46,  # wine glass       (kitchen)
    # 47,  # cup              (kitchen)
    # 48,  # fork             (kitchen)
    # 49,  # knife            (kitchen)
    # 50,  # spoon            (kitchen)
    # 51,  # bowl             (kitchen)
    # 52,  # banana           (food)
    # 53,  # apple            (food)
    # 54,  # sandwich         (food)
    # 55,  # orange           (food)
    # 56,  # broccoli         (food)
    # 57,  # carrot           (food)
    # 58,  # hot dog          (food)
    # 59,  # pizza            (food)
    # 60,  # donut            (food)
    # 61,  # cake             (food)
    62,  # chair            (furniture)
    63,  # couch            (furniture)
    64,  # potted plant     (furniture)
    65,  # bed              (furniture)
    67,  # dining table     (furniture)
    70,  # toilet           (furniture)
    # 72,  # tv               (electronic)
    # 73,  # laptop           (electronic)
    # 74,  # mouse            (electronic)
    # 75,  # remote           (electronic)
    # 76,  # keyboard         (electronic)
    # 77,  # cell phone       (electronic)
    # 78,  # microwave        (appliance)
    # 79,  # oven             (appliance)
    # 80,  # toaster          (appliance)
    # 81,  # sink             (appliance)
    # 82,  # refrigerator     (appliance)
    # 84,  # book             (indoor)
    # 85,  # clock            (indoor)
    # 86,  # vase             (indoor)
    # 87,  # scissors         (indoor)
    # 88,  # teddy bear       (indoor)
    # 89,  # hair drier       (indoor)
    # 90,  # toothbrush       (indoor)
}


def process(img_id: int, img_dir: str, coco: COCO, coco_ori: COCO, seed: int = 0):
    L.seed_everything(seed)

    # load an image and annotations
    img_path = os.path.join(img_dir, coco.imgs[img_id]["file_name"])
    img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = torch.tensor(img_np, dtype=torch.float32, device=device).permute(2, 0, 1) / 255
    height, width, _ = img_np.shape

    # load annotations
    coco_annot_list = [ann for ann in coco.imgToAnns[img_id] if ann["iscrowd"] == 0]

    # count objects
    obj_counts = defaultdict(int)
    unoccluded_annot_ids = set()
    for ann in coco_annot_list:
        obj_id, duplicate_id = divmod(ann["id"], LABEL_OFFSET)
        obj_counts[obj_id] += 1
        if duplicate_id > 0:
            unoccluded_annot_ids.add(obj_id)

    # decrement the obj count
    obj_counts = {obj_id: obj_cnt - 1 for obj_id, obj_cnt in obj_counts.items()}

    # extract masks
    obj_masks = {}
    obj_areas = []
    for obj_id in obj_counts.keys():
        msk_np = coco_ori.annToMask(coco_ori.anns[obj_id])
        if msk_np.sum() < 1024:  # COCO small masks are better not to be refined
            mask = torch.tensor(msk_np, dtype=torch.float32, device=device)
        else:
            msk_trimap_np = np.where(
                msk_np > 0.5,
                255,
                np.where(cv2.dilate(msk_np, np.ones((5, 5))) > 0.5, 128, 0),
            )
            with torch.inference_mode():
                pixel_values = vitmatte_processor(
                    images=img_np, trimaps=msk_trimap_np, return_tensors="pt"
                ).pixel_values
                matting_outputs = vitmatte(pixel_values.to(device))
                mask = matting_outputs.alphas[..., :height, :width].reshape(height, width)

        if torch.all(mask < 0.5):
            continue
        obj_masks[obj_id] = mask
        obj_areas.append((mask.sum(), obj_id))
    obj_areas.sort(reverse=True)

    # initialize the labelmap
    labelmap = torch.full((height, width), -1, dtype=torch.int64, device=device)
    for obj_area, obj_id in obj_areas:
        mask = obj_masks[obj_id]
        labelmap[mask > 0.5] = obj_id

    # prepare an obj pool
    obj_pool = []
    for obj_id, obj_cnt in obj_counts.items():
        obj_pool.extend([obj_id] * obj_cnt)
    random.shuffle(obj_pool)
    obj_num = len(obj_pool)

    # final outputs
    composed_img = img.permute(1, 2, 0).clone()
    composed_mask = torch.zeros((height, width), dtype=torch.float32, device=device)
    composed_label = labelmap * LABEL_OFFSET
    rendered_img = torch.zeros((obj_num, height, width, 3), dtype=torch.float32, device=device)
    rendered_mask = torch.zeros((obj_num, height, width), dtype=torch.float32, device=device)

    # compose
    for i, obj_id in enumerate(obj_pool):
        # define mask
        mask = obj_masks[obj_id]
        x1, y1, x2, y2 = torchvision.ops.masks_to_boxes(mask[None] > 0.5)[0].long()
        mask = mask[y1:y2, x1:x2]
        obj_img_mask = torch.cat([img[:, y1:y2, x1:x2], mask[None]], dim=0)

        # scale and shift the mask
        scale_factor = random.uniform(*TARGET_SCALE_REL_RANGE)
        if (x2 - x1) * scale_factor >= 1 and (y2 - y1) * scale_factor >= 1:
            obj_img_mask = F.interpolate(
                obj_img_mask.unsqueeze(0), scale_factor=scale_factor, mode="bilinear"
            ).squeeze(0)
        else:
            pass
        _, mask_height, mask_width = obj_img_mask.shape

        # define the position
        top = random.randint(-mask_height // 2, height - mask_height // 2)
        left = random.randint(-mask_width // 2, width - mask_width // 2)

        patch_top = top
        patch_bottom = top + mask_height
        patch_left = left
        patch_right = left + mask_width

        canvas_top = max(0, patch_top)
        canvas_bottom = min(height, patch_bottom)
        canvas_left = max(0, patch_left)
        canvas_right = min(width, patch_right)

        patch_sub_top = canvas_top - patch_top
        patch_sub_bottom = canvas_bottom - patch_top
        patch_sub_left = canvas_left - patch_left
        patch_sub_right = canvas_right - patch_left

        rendered_img[i, canvas_top:canvas_bottom, canvas_left:canvas_right, :] = obj_img_mask[
            :3, patch_sub_top:patch_sub_bottom, patch_sub_left:patch_sub_right
        ].permute(1, 2, 0)
        rendered_mask[i, canvas_top:canvas_bottom, canvas_left:canvas_right] = obj_img_mask[
            3, patch_sub_top:patch_sub_bottom, patch_sub_left:patch_sub_right
        ]

        # color augmentation
        rendered_img[i] = change_object_color_hsv(rendered_img[i][None], rendered_mask[i][None])[0]

        # compose
        composed_img = (
            composed_img * (1 - rendered_mask[i, :, :, None])
            + rendered_img[i] * rendered_mask[i, :, :, None]
        )
        composed_mask = torch.max(composed_mask, rendered_mask[i])
        composed_label[rendered_mask[i] > 0.5] = obj_id * LABEL_OFFSET + i + 1

    return (
        composed_img,
        composed_mask,
        composed_label,
        rendered_img,
        rendered_mask,
        unoccluded_annot_ids,
    )


def get_contour_bbox_area(
    mask: Float[torch.Tensor, "n h w"],
) -> tuple[list[list[list[int]]], Float[np.ndarray, "n 4"], Int64[np.ndarray, " n"]]:
    if len(mask.shape) == 2:
        squeeze_needed = True
        mask = mask[None]
    else:
        squeeze_needed = False

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    num, height, width = mask.shape  # mask must be three channels
    mask = (mask > 0.5).transpose(1, 2, 0)  # just in case & channel-last

    # add border to make find_contour work correctly
    mask_bordered = np.zeros((height + 2, width + 2, num), dtype=bool)
    mask_bordered[1:-1, 1:-1, :] = mask

    # coco format -> bbox
    fortran_ground_truth_binary_mask = np.asfortranarray(mask_bordered)
    encoded_ground_truth = pycocotools.mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = pycocotools.mask.area(encoded_ground_truth)
    bboxes_xywh = pycocotools.mask.toBbox(encoded_ground_truth)  # (num, 4)
    bboxes_xywh[:, :2] -= 1  # to compensate for padding

    # find contours
    segmentations = []
    for n in range(num):
        contours = find_contours(mask_bordered[:, :, n], 0.5)
        segm_per_img = []
        for contour in contours:  # for each connected component
            contour = (
                np.flip(contour, axis=1) - 1
            )  # (B, ij) -> (B, xy), and compensate for padding
            contour_ravel = contour.ravel().tolist()
            segm_per_img.append(contour_ravel)
        segmentations.append(segm_per_img)

    if squeeze_needed:
        return segmentations[0], bboxes_xywh[0].tolist(), ground_truth_area.tolist()[0]
    else:
        return segmentations, bboxes_xywh.tolist(), ground_truth_area.tolist()


def register_composed_result(
    coco_ori: COCO,
    img_id: int,
    composed_img: Float[Tensor, "h w 3"],
    composed_mask: Float[Tensor, "h w"],
    composed_label: Int64[Tensor, "h w"],
    rendered_img: Float[Tensor, "b h w 3"],
    rendered_mask: Float[Tensor, "b h w"],
    unoccluded_annot_ids: list[int],
    label_offset: int = 1000,
) -> tuple[list[dict[str, Any]], int]:
    obj_batch, height, width, channel = rendered_img.shape
    assert channel == 3
    assert rendered_mask.shape == (obj_batch, height, width)
    assert composed_img.shape == (height, width, 3)
    assert composed_mask.shape == (height, width)
    assert composed_label.shape == (height, width)

    annotations = {ann["id"]: ann for ann in coco_ori.imgToAnns[img_id]}
    all_annot_ids = annotations.keys()
    new_annot_list_per_img = []
    composed_obj_num = 0

    for lbl_id in torch.unique(composed_label):
        lbl_id = lbl_id.item()
        # lbl_id = -1 is undefined
        if lbl_id < 0:
            continue

        ann_id, composite_id = divmod(lbl_id, label_offset)
        composite_id -= 1  # NOTE: the composition 'residual' labels are 1-indexed, so we need to decrement here
        assert (
            ann_id in all_annot_ids
        ), f"{torch.unique(composed_label)=}, {ann_id=}, {all_annot_ids=}"
        ann = annotations[ann_id]
        msk = composed_label == lbl_id

        # existing mask
        if composite_id < 0:  # NOTE: composite_id is already decremented
            visible_segm, visible_bbox, visible_area = get_contour_bbox_area(msk)
            amodal_segm, amodal_bbox, amodal_area = ann["segmentation"], ann["bbox"], ann["area"]
            mask_type = "real_whole" if (ann_id in unoccluded_annot_ids) else "real_partial"
        # new mask
        else:
            visible_segm, visible_bbox, visible_area = get_contour_bbox_area(msk)
            amodal_segm, amodal_bbox, amodal_area = get_contour_bbox_area(
                rendered_mask[composite_id]
            )
            mask_type = "syn"
            composed_obj_num += 1

        # write down to dict
        new_ann = {
            "amodal_bbox": amodal_bbox,
            "amodal_segm": amodal_segm,
            "amodal_area": amodal_area,
            "visible_bbox": visible_bbox,
            "visible_segm": visible_segm,
            "visible_area": visible_area,
            "background_objs_segm": [],
            "occluder_segm": [],
            "bbox": amodal_bbox,
            "segmentation": amodal_segm,
            "area": amodal_area,
            "iscrowd": False,
            "id": lbl_id,  # USE OFFSETTED ANNOT_ID AS A NEW ID!!!
            "image_id": img_id,
            "category_id": ann["category_id"],
            "mask_type": mask_type,
        }
        new_annot_list_per_img.append(new_ann)

    return new_annot_list_per_img, composed_obj_num


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
    image_save_dir = os.path.join(coco_root_dir, f"{is_train_or_val}2017_composed_simple")

    img_dir = os.path.join(coco_root_dir, f"{is_train_or_val}2017")
    coco = COCO(
        os.path.join(
            coco_root_dir,
            f"annotations/instances_{is_train_or_val}2017_kakuda_composition_labels_refactored.json",
        )
    )
    coco_ori = COCO(
        os.path.join(coco_root_dir, f"annotations/instances_{is_train_or_val}2017.json")
    )
    annot_id_list = coco.getAnnIds(catIds=[])

    new_img_list = []
    new_img_set = set()
    new_annot_list = []

    # load json if exists (assuming the process was interrupted in the middle)
    json_outpath = os.path.join(
        coco_root_dir,
        f"annotations/instances_{is_train_or_val}2017_kakuda_composition_labels_simple.json",
    )
    if os.path.isfile(json_outpath):
        dprint(
            f"[main] Loading {json_outpath}, assuming that the process was interrupted in the middle."
        )
        new_img_list, new_annot_list = load_json(json_outpath)
        new_img_set = {int(img_meta["id"]) for img_meta in new_img_list}
    elif os.path.isdir(image_save_dir) and os.listdir(image_save_dir):
        print(
            f"[main] Manually remove {image_save_dir} first!!! (since {json_outpath} does not exist)"
        )
        exit()
    os.makedirs(image_save_dir, exist_ok=True)

    for img_id in tqdm(coco.getImgIds()):

        if img_id in new_img_set:
            continue

        # compose
        new_annots_per_img = []

        (
            composed_img,
            composed_mask,
            composed_label,
            homography_rendered_img,
            homography_rendered_mask,
            unoccluded_annot_ids,
        ) = process(img_id, img_dir, coco, coco_ori, seed=0)

        # register
        new_annots_per_img, composed_obj_num = register_composed_result(
            coco_ori,
            img_id,
            composed_img,
            composed_mask,
            composed_label,
            homography_rendered_img,
            homography_rendered_mask,
            unoccluded_annot_ids,
            label_offset=LABEL_OFFSET,
        )

        new_annot_list.extend(new_annots_per_img)

        # save the image (NOTE: amodal texture is so far not saved!!!)
        image_save_path = os.path.join(image_save_dir, coco.imgs[img_id]["file_name"])
        ret_uint8 = np.clip(255 * composed_img.cpu().numpy(), 0, 255).astype(np.uint8)
        cv2.imwrite(image_save_path, ret_uint8[:, :, ::-1])
        new_img_list.append(coco.imgs[img_id])
        new_img_set.add(img_id)
        dprint(f"\n{img_id}: saved! (Composed object num: {composed_obj_num})\n")

        if new_img_list and len(new_img_list) % JSON_SAVE_FREQ == 0:
            save_json(new_img_list, new_annot_list, json_outpath, coco)

    # save into a json file
    save_json(new_img_list, new_annot_list, json_outpath, coco)
    print(f"Finished! {json_outpath=}")
