import glob
import json
import os
import random
import sys
from typing import Any, Optional

import cv2
import depth_pro
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pycocotools
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from jaxtyping import Float, Int64, UInt8
from kornia.color import hsv_to_rgb, rgb_to_hsv
from kornia.contrib import connected_components
from kornia.filters import box_blur, canny
from kornia.geometry.homography import find_homography_dlt
from kornia.geometry.transform import warp_perspective
from kornia.morphology import closing, dilation, erosion, opening
from pycocotools.coco import COCO
from pytorch3d.ops import knn_points
from pytorch3d.transforms import axis_angle_to_matrix
from skimage.measure import find_contours
from skimage.transform import resize
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
    VitMatteForImageMatting,
    VitMatteImageProcessor,
)

from geocalib import GeoCalib

sys.path.append("../")
from third_party.MoGe.moge.model import MoGeModel

device = "cuda:1"

# DepthPro
depthpro_cfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
depthpro_cfg.checkpoint_uri = "../../../github/ml-depth-pro/checkpoints/depth_pro.pt"
depthpro, depthpro_transform = depth_pro.create_model_and_transforms(depthpro_cfg)
depthpro.eval().to(device)

# OneFormer
ONEFORMER_ADE20K = True
USE_NATTEN = False
if ONEFORMER_ADE20K:
    if USE_NATTEN:
        oneformer_processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_dinat_large"
        )
        oneformer = (
            OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_dinat_large"
            )
            .eval()
            .to(device)
        )
    else:
        oneformer_processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large"
        )
        oneformer = (
            OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large"
            )
            .eval()
            .to(device)
        )

    oneformer_task_inputs = oneformer_processor._preprocess_text(["the task is semantic"])
    oneformer_ground_labels = torch.tensor(
        [
            3,  # 3 => floor
            6,  # 6 => road, route
            9,  # 9 => grass
            11,  # 11 => sidewalk, pavement
            13,  # 13 => earth, ground
            28,  # 28 => rug
            46,  # 46 => sand  #######################################################################
            53,  # 53 => stairs
            59,  # 59 => stairway, staircase
            96,  # 96 => escalator, moving staircase, moving stairway
            121,  # 121 => step, stair
        ],
        dtype=torch.int64,
        device=device,
    )
    oneformer_sky_labels = torch.tensor([2], dtype=torch.int64, device=device)
else:
    raise NotImplementedError("Not ready for COCO yet")

# Metric depth
geocalib = GeoCalib().eval().to(device)
metric3dv2 = (
    torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True).eval().to(device)
)
moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").eval().to(device)

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
COMPOSITION_RETRY_NUM = 3
MAX_OBJECT_NUM = 32
TARGET_DEPTH_REL_RANGE = (0.75, 2.5)

ABOVE_GROUND_DIST_THRESH = 0.1
COLLISION_DIST_THRESH = 0.01
COLLISION_RATIO_THRESH = 0.01

LABEL_OFFSET = 1000

JSON_SAVE_FREQ = 1000

VERBOSE = False


# In[2]:


class CompositionError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def dprint(*arg):
    if VERBOSE:
        print(*arg, flush=True)
    return


# In[3]:


def PCA(
    data: Float[Tensor, "b n m"],
    weights: Float[torch.Tensor, "b n"] = None,
    correlation: bool = False,
    sort: bool = True,
) -> tuple[Float[Tensor, "b m"], Float[Tensor, "b m m"]]:
    """Applies Batch PCA to input tensor.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of shape (B, N, M)
        B: batch size, N: number of records, M: number of features

    weights : torch.Tensor, optional
        Input tensor of shape (B, N) containing weights for each record.

    correlation : bool, optional
        If True, compute correlation matrix. If False, compute covariance matrix.

    sort : bool, optional
        If True, sort eigenvalues/vectors in descending order.

    Returns
    -------
    eigenvalues : torch.Tensor
        Eigenvalues of shape (B, M)

    eigenvectors : torch.Tensor
        Eigenvectors of shape (B, M, M)
    """
    assert torch.isfinite(data).all()
    B, N, D = data.shape
    if weights is not None:
        dprint("[PCA] weight applied")
        assert weights.shape == (B, N), f"Invalid weights shape: {weights.shape} ({data.shape=})"
        assert torch.all(weights >= 0), "Weights must be non-negative."

    # Subtract mean along record dimension
    if weights is not None:
        mean = (data * weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / weights.sum(
            dim=1, keepdim=True
        )
    else:
        mean = data.mean(dim=1, keepdim=True)
    data_adjusted = data - mean

    # Compute matrix based on correlation or covariance
    if correlation:
        if weights is not None:
            raise NotImplementedError("Weighted correlation matrix is not implemented.")
        # Compute correlation for each batch
        matrix = torch.stack([torch.corrcoef(batch_data.T) for batch_data in data_adjusted])
    else:
        # https://stackoverflow.com/questions/71357619/how-do-i-compute-batched-sample-covariance-in-pytorch
        def batch_cov(points, weights):
            B, N, D = points.size()

            if weights is not None:
                weights = weights / (
                    1e-6 + weights.sum(dim=1, keepdim=True)
                )  # Normalize weights so they sum to 1 along the N dim

                mean = (points * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
                diffs = (points - mean).reshape(B * N, D)
                prods = torch.bmm(
                    diffs.unsqueeze(2) * weights.reshape(B * N, 1, 1), diffs.unsqueeze(1)
                ).reshape(B, N, D, D)
                bcov = prods.sum(
                    dim=1
                )  # Note that we don't need to divide by N-1 because the weights sum to one.
            else:
                mean = points.mean(dim=1).unsqueeze(1)
                diffs = (points - mean).reshape(B * N, D)
                prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
                bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate

            return bcov  # (B, D, D)

        matrix = batch_cov(data_adjusted, weights)

    # Compute eigenvalues and eigenvectors for each batch matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    # Sort if required (in descending order)
    if sort:
        # Flip eigenvalues and eigenvectors to descending order
        sorted_indices = eigenvalues.argsort(dim=1, descending=True)
        batch_indices = torch.arange(eigenvalues.size(0))[:, None]

        eigenvalues = eigenvalues[batch_indices, sorted_indices]
        eigenvectors = eigenvectors[batch_indices, :, sorted_indices]

    return eigenvalues, eigenvectors.permute(0, 2, 1)


def best_fitting_plane(
    points: Float[torch.Tensor, "b n 3"],
    weights: Float[torch.Tensor, "b n"] = None,
) -> tuple[Float[torch.Tensor, "b 4"], Float[torch.Tensor, " b"]]:
    """Computes the best fitting plane for batched points.

    Parameters
    ----------
    points : torch.Tensor
        Input tensor of shape (B, N, 3)
        B: batch size, N: number of points, 3: x,y,z coordinates

    weights : torch.Tensor, optional
        Input tensor of shape (B, N) containing weights for each point.

    equation : bool, optional
        If True, return plane coefficients.
        If False, return point and normal vector.

    Returns
    -------
        [a, b, c, d] : torch.Tensor of shape (B, 4)
        error : torch.Tensor of shape (B,)
    """
    batch, num, channel = points.shape
    assert channel == 3
    assert torch.isfinite(points).all()
    if weights is not None:
        assert weights.shape == (
            batch,
            num,
        ), f"Invalid weights shape: {weights.shape} ({points.shape=})"
        assert torch.all(weights >= 0), "Weights must be non-negative."

    # Compute PCA for each batch of points
    try:
        eigenvalues, eigenvectors = PCA(points, weights)
    except torch._C._LinAlgError as e:
        raise CompositionError(
            f"[best_fitting_plane] torch.linalg.eigh didn't converge in PCA: {e}"
        )

    # The normal is the last eigenvector (smallest eigenvalue)
    normal = eigenvectors[:, :, 2]

    # Get mean point for each batch
    if weights is not None:
        center_point = (points * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        )
        assert torch.isfinite(input=center_point).all(), f"{weights.sum(dim=1, keepdim=True)=}"
    else:
        center_point = points.mean(dim=1)

    # Compute plane equation coefficients
    a, b, c = normal.T
    d = -(normal * center_point).sum(dim=1)

    # normalize the normal vector
    nunom = (
        a.reshape(-1, 1) * points[:, :, 0]
        + b.reshape(-1, 1) * points[:, :, 1]
        + c.reshape(-1, 1) * points[:, :, 2]
        + d.reshape(-1, 1)
    )
    denom = torch.sqrt(a * a + b * b + c * c) + 1e-6
    error = (nunom / (denom + 1e-6)).mean(dim=1)

    return torch.stack([a, b, c, d], dim=-1) / denom, error


# In[4]:


def infer_metric_points_and_focal(
    img: Float[Tensor, "3 h w"]
) -> tuple[Float[Tensor, "h w 3"], Float[Tensor, "3 3"]]:
    channel, height, width = img.shape
    assert channel == 3

    def unproject(
        depth: Float[Tensor, "h w"], f_px: float, height: int, width: int
    ) -> Float[Tensor, "h w"]:
        y, x = torch.meshgrid(
            torch.arange(height, dtype=depth.dtype, device=depth.device),
            torch.arange(width, dtype=depth.dtype, device=depth.device),
            indexing="ij",
        )
        x = (x - width / 2) * depth / f_px
        y = (y - height / 2) * depth / f_px
        return torch.stack([x, y, depth], dim=-1)

    # DepthPro
    if True:
        prediction = depthpro.infer(2 * img.unsqueeze(0) - 1, f_px=None)
        depth = prediction["depth"]  # Depth in [m].
        f_px = prediction["focallength_px"]
        points = unproject(depth, f_px, height, width)
        intrinsic = torch.diag(torch.tensor([f_px, f_px, 1], dtype=img.dtype, device=img.device))
        intrinsic[0, 2] = width / 2
        intrinsic[1, 2] = height / 2
    else:
        raise NotImplementedError
    return points.reshape(height, width, 3), intrinsic


def infer_semantic_mask(img: Float[Tensor, "3 h w"]) -> Float[Tensor, "h w"]:
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    b, c, h_ori, w_ori = img.shape
    assert c == 3

    mean = torch.tensor(
        oneformer_processor.image_processor.image_mean,
        dtype=img.dtype,
        device=img.device,
    ).reshape(1, 3, 1, 1)

    std = torch.tensor(
        oneformer_processor.image_processor.image_std,
        dtype=img.dtype,
        device=img.device,
    ).reshape(1, 3, 1, 1)

    scale_factor = oneformer_processor.image_processor.size["shortest_edge"] / min(img.shape[-2:])
    pixel_values = F.interpolate(img, scale_factor=scale_factor, mode="bilinear")
    pixel_values *= 255 * oneformer_processor.image_processor.rescale_factor
    pixel_values = (pixel_values - mean) / std

    b, c, h, w = pixel_values.shape
    semantic_inputs = {
        "pixel_values": pixel_values,
        "pixel_mask": torch.ones(b, h, w, dtype=torch.int64, device=img.device),
        "task_inputs": oneformer_task_inputs.repeat(b, 1).to(img.device),
    }
    semantic_outputs = oneformer(**semantic_inputs)
    predicted_semantic_map = oneformer_processor.post_process_semantic_segmentation(
        semantic_outputs,
        target_sizes=[(h_ori, w_ori)] * b,
    )
    predicted_semantic_map = torch.stack(predicted_semantic_map, dim=0)  # (b, h_ori, w_ori)

    return predicted_semantic_map.squeeze()


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


def denoise_disconnected_obj_points(
    obj_masks: Float[torch.Tensor, "b h w"],
    obj_points: Float[torch.Tensor, "(b) h w 3"],
    knn_num: int = 10,
    knn_threshold: float = 0.1,
    remove_isolated_components: bool = True,
    apply_closing: bool = True,
) -> Float[Tensor, "b h w"]:
    obj_batch, height, width = obj_masks.shape
    assert obj_points.shape == (height, width, 3) or obj_points.shape == (
        obj_batch,
        height,
        width,
        3,
    )
    if len(obj_points.shape) == 3:
        obj_points = obj_points[None].expand(obj_batch, -1, -1, -1)

    obj_points_list = []
    obj_points_len = []
    for pts, msk in zip(obj_points, obj_masks):
        pts = pts[msk > 0.5]
        obj_points_list.append(pts)
        obj_points_len.append(len(pts))

    # Pad the tensors to the same length
    obj_points_packed = pad_sequence(obj_points_list, batch_first=True)
    obj_points_len = torch.tensor(obj_points_len, dtype=torch.long, device=obj_masks.device)

    nn_dists, nn_idx, nn = knn_points(
        obj_points_packed, obj_points_packed, obj_points_len, obj_points_len, K=knn_num + 1
    )

    obj_masks_denoised = obj_masks.clone()
    for b in range(obj_batch):
        # remove isolated points
        point_num = obj_points_len[b]
        msk = obj_masks[b]
        knn_map = torch.ones_like(obj_masks[b])
        knn_map[msk > 0.5] = nn_dists[0, :point_num, 1:].mean(dim=1)
        obj_masks_denoised[b, knn_map > knn_threshold**2] = 0

        # discard isorated clusters (NOTE: assuming that the original mask is simply connected)
        if remove_isolated_components:
            component_labelmap = connected_components(
                (obj_masks_denoised[b] > 0.5).float().reshape(1, 1, height, width),
                num_iterations=1000,
            ).reshape(height, width)
            component_labels = torch.sort(torch.unique(component_labelmap)).values
            component_labelmap = torch.searchsorted(
                component_labels, input=component_labelmap
            )  # make the label consecutive
            component_areas = torch.bincount(component_labelmap.reshape(-1))
            if len(component_areas) == 1:
                raise CompositionError(
                    "[denoise_disconnected_obj_points] Object disappeared after denoising"
                )
            major_component = component_labelmap == 1 + component_areas[1:].argmax()
            obj_masks_denoised[b] *= major_component

    # fill in tiny holes inside objects
    if apply_closing:
        closing_kernel = torch.ones(3, 3, dtype=obj_masks.dtype, device=obj_masks.device)
        obj_masks_denoised = closing(
            obj_masks_denoised.reshape(obj_batch, 1, height, width), closing_kernel
        ).reshape(obj_batch, height, width)

    return obj_masks_denoised


def is_object_on(
    object_points: Float[Tensor, "n 3"],
    support_points: Float[Tensor, "m 3"],
    local_support_plane_fitting: bool = True,
    near_distance_thresh: float = 0.1,
) -> tuple[bool, Float[Tensor, "4"]]:
    assert len(object_points.shape) == 2 and object_points.shape[1] == 3, f"{object_points.shape=}"
    assert (
        len(support_points.shape) == 2 and support_points.shape[1] == 3
    ), f"{support_points.shape=}"

    if local_support_plane_fitting:
        object_center = object_points.mean(dim=0, keepdim=True)
        plane_fitting_weight = 1 / torch.square(support_points - object_center).sum(dim=-1).clip(
            1, None
        )

        plane_coeffs, plane_fitting_error = best_fitting_plane(
            points=support_points.reshape(1, -1, 3),
            weights=plane_fitting_weight.reshape(1, -1),
        )
        assert torch.isfinite(plane_coeffs).all(), f"{plane_coeffs=}"

        # make pb always positive (plane_coeffs = [pa, pb, pc, pd])
        pb_sign = torch.sign(plane_coeffs[:, 1])
        plane_coeffs = plane_coeffs * torch.where(pb_sign == 0, torch.ones_like(pb_sign), pb_sign)

        # get the distance from the object points to the ground
        object_points_homogeneous = torch.cat(
            [object_points, torch.ones_like(object_points[:, 0:1])], dim=-1
        )
        dists = (plane_coeffs * object_points_homogeneous).sum(dim=-1).abs() / (
            torch.norm(plane_coeffs[:, :3], dim=-1) + 1e-6
        )
        assert dists.shape == (object_points.shape[0],), f"{dists.shape=}"
        dprint(f"[is_object_on] Distance from the support: {dists.min().item():.6f}")

        object_is_on_the_support = dists.min() < near_distance_thresh
        return object_is_on_the_support.reshape(1), plane_coeffs.reshape(4)

    else:
        raise NotImplementedError("[is_object_on] KNN not yet implemented")
        return False, torch.full((4,), torch.nan, device=object_points.device)


def is_plane_orthogonal_to_gravity(
    plane_normal: Float[Tensor, "3"],
    gravity_direction: Float[Tensor, "3"],
    cosine_threshold: float = 0.95,
):
    assert plane_normal.shape == gravity_direction.shape == (3,)
    cos_sim = torch.sum(
        F.normalize(plane_normal, dim=0) * F.normalize(gravity_direction, dim=0)
    ).abs()
    dprint(f"[is_plane_orthogonal_to_gravity] {cos_sim.item()=}")
    return cos_sim > cosine_threshold


def is_object_planar(
    object_points: Float[Tensor, "n 3"],
    plane_fitting_depth_margin: float = 0.2,
):
    assert len(object_points.shape) == 2 and object_points.shape[1] == 3, f"{object_points.shape=}"

    object_depths = object_points[:, 2]
    object_depth_median = torch.median(object_depths)
    object_points_for_fitting = object_points[
        (object_depth_median - plane_fitting_depth_margin < object_depths)
        * (object_depths < object_depth_median + plane_fitting_depth_margin)
    ]

    plane_fitting_weight = 1 / torch.square(
        object_points_for_fitting[:, 2].reshape(1, -1) - object_depth_median
    ).clip(0.0001, None)
    plane_coeffs, plane_fitting_error = best_fitting_plane(
        points=object_points_for_fitting.reshape(1, -1, 3),
        weights=plane_fitting_weight.reshape(1, -1),
    )
    assert torch.isfinite(plane_coeffs).all(), f"{plane_coeffs=}"

    # make pc always positive (plane_coeffs = [pa, pb, pc, pd])
    pc_sign = torch.sign(plane_coeffs[:, 2])
    plane_coeffs = plane_coeffs * torch.where(pc_sign == 0, torch.ones_like(pc_sign), pc_sign)

    is_planar = plane_coeffs[:, 2].abs() > torch.max(
        plane_coeffs[:, 0].abs(), plane_coeffs[:, 1].abs()
    )
    return is_planar.reshape(1), plane_coeffs.reshape(4)


# In[5]:


def warp_points_at_random(
    points: Float[Tensor, "h w 3"],
    depths: Float[Tensor, "h w"],
    obj_masks: Float[Tensor, "b h w"],
    ground_mask: Float[Tensor, "h w"],
    ground_plane_coeff: Float[Tensor, "4"],
    intrinsics: Float[Tensor, "3 3"],
    tgt_depth_rel_range: tuple[float, float] = (0.75, 2.5),
) -> tuple[Float[Tensor, "b h w 3"], Float[Tensor, "b 3 3"], Float[Tensor, "b 3"]]:
    # sanity check
    obj_batch, height, width = obj_masks.shape
    assert points.shape == (height, width, 3)
    assert depths.shape == (height, width)
    assert ground_mask.shape == (height, width)
    assert ground_plane_coeff.shape == (4,)
    assert intrinsics.shape == (3, 3)
    tgt_depth_rel_min, tgt_depth_rel_max = tgt_depth_rel_range
    assert tgt_depth_rel_min < tgt_depth_rel_max

    assert torch.isfinite(ground_plane_coeff).all()

    # add bkg_batch dimension (for historical reasons)
    bkg_batch = 1
    points = points.reshape(bkg_batch, height, width, 3)
    depths = depths.reshape(bkg_batch, height, width)
    ground_mask = ground_mask.reshape(bkg_batch, height, width)
    ground_plane_coeff = ground_plane_coeff.reshape(bkg_batch, 4)
    intrinsics = intrinsics.reshape(bkg_batch, 3, 3)

    # On a rare occasion, the vertically lowest border between the ground and the mask can be higher than the lowest mask pixel
    # (e.g. a man on a motorbike, the toe is inside the motorbike mask and the thigh contacts the 2D ground mask)
    # So, it may be enough to pick the lowest mask point without looking at the ground mask (or throwing an error is another possibility)
    mask_ground_intersection_binary = obj_masks.reshape(bkg_batch, -1, height, width)

    # search for the source foot positions
    mask_ground_intersection_vertical = torch.any(mask_ground_intersection_binary, dim=-1)
    _, mask_ground_intersection_vertical_idx, mask_ground_intersection_vertical_px = (
        mask_ground_intersection_vertical.nonzero(as_tuple=True)
    )
    foot_pix_src_y = mask_ground_intersection_vertical_px[
        mask_ground_intersection_vertical_idx.bincount().cumsum(dim=0) - 1
    ]  # lowest pixel

    mask_ground_intersection_horizontal = torch.any(mask_ground_intersection_binary, dim=-2)
    _, mask_horizontal_idx, mask_horizontal_px = mask_ground_intersection_horizontal.nonzero(
        as_tuple=True
    )
    foot_pix_src_x = torch.bincount(
        mask_horizontal_idx, weights=mask_horizontal_px
    ) / torch.bincount(mask_horizontal_idx)
    foot_pix_src = torch.stack(
        [foot_pix_src_x, foot_pix_src_y, torch.ones_like(foot_pix_src_x)], dim=-1
    )
    foot_pos_src_z = depths[:, foot_pix_src_y, foot_pix_src_x.int()]
    foot_pos_src: torch.Tensor = (
        torch.linalg.inv(intrinsics.float()) @ foot_pix_src.reshape(-1, 3, 1).float()
    ).reshape(bkg_batch, obj_batch, 3) * foot_pos_src_z.reshape(bkg_batch, obj_batch, 1)
    assert foot_pos_src.shape == (bkg_batch, obj_batch, 3)

    # target foot position
    tgt_depth_min = foot_pos_src_z * tgt_depth_rel_min
    tgt_depth_max = torch.minimum(
        foot_pos_src_z * tgt_depth_rel_max, depths.flatten(-2).max(dim=-1).values
    )
    foot_pos_tgt_z = tgt_depth_min + torch.rand_like(foot_pos_src_z) * (
        tgt_depth_max - tgt_depth_min
    )

    foot_pos_tgt_x_max = (width / 2) * foot_pos_tgt_z / intrinsics[:, 0, 0].reshape(bkg_batch, 1)
    foot_pos_tgt_x = (torch.rand_like(foot_pos_tgt_z) * 2 - 1) * foot_pos_tgt_x_max

    PA, PB, PC, PD = torch.split(ground_plane_coeff, 1, dim=-1)
    foot_pos_tgt_y = (-PD - PC * foot_pos_tgt_z - PA * foot_pos_tgt_x) / PB
    foot_pos_tgt = torch.stack([foot_pos_tgt_x, foot_pos_tgt_y, foot_pos_tgt_z], dim=-1)
    assert foot_pos_tgt.shape == (bkg_batch, obj_batch, 3)

    # place foot_tgt on the ground_mask if visible (by project & unproject)
    foot_pix_tgt = (
        intrinsics.reshape(bkg_batch, 1, 3, 3) @ foot_pos_tgt.reshape(bkg_batch, obj_batch, 3, 1)
    ).reshape(bkg_batch, obj_batch, 3)
    foot_pix_tgt = foot_pix_tgt[:, :, :2] / (foot_pix_tgt[:, :, 2:3] + 1e-6)
    foot_tgt_visible = (
        (0 <= foot_pix_tgt[:, :, 0])
        * (foot_pix_tgt[:, :, 0] < width)
        * (0 <= foot_pix_tgt[:, :, 1])
        * (foot_pix_tgt[:, :, 1] < height)
    )
    foot_pix_tgt = foot_pix_tgt.long()
    foot_pix_tgt[:, :, 0] = foot_pix_tgt[:, :, 0].clip(0, width - 1)
    foot_pix_tgt[:, :, 1] = foot_pix_tgt[:, :, 1].clip(0, height - 1)
    foot_depth_tgt = depths[
        torch.arange(bkg_batch).unsqueeze(1).repeat(1, obj_batch),
        foot_pix_tgt[:, :, 1],
        foot_pix_tgt[:, :, 0],
    ]
    foot_pos_tgt = torch.where(
        foot_tgt_visible * (foot_depth_tgt > foot_pos_tgt[:, :, 2]),
        foot_pos_tgt * foot_depth_tgt / foot_pos_tgt[:, :, 2],
        foot_pos_tgt,
    )

    # define warp (rotation & shift ON THE GROUND PLANE)
    assert torch.allclose(PA * PA + PB * PB + PC * PC, torch.ones(1, device=PA.device))
    ground_origin = ground_plane_coeff[:, :3] * (-PD)
    origin_to_src = F.normalize(foot_pos_src - ground_origin, dim=-1)
    origin_to_tgt = F.normalize(foot_pos_tgt - ground_origin, dim=-1)
    rot_axis = torch.linalg.cross(
        origin_to_src, origin_to_tgt
    )  # NOTE: should be the same as ground_plane_coeff[:, :3] removing scale diff
    clockwise = torch.sum(rot_axis * ground_plane_coeff[:, :3], dim=-1).sign()
    eps = 1e-7
    rot_angle = (
        torch.arccos(torch.sum(origin_to_src * origin_to_tgt, dim=-1).clip(-1 + eps, 1 - eps))
        * clockwise
    )
    rotmat = axis_angle_to_matrix(
        rot_angle.reshape(bkg_batch, obj_batch, 1)
        * ground_plane_coeff[:, :3].reshape(bkg_batch, 1, 3)
    )
    shift = foot_pos_tgt - (rotmat @ foot_pos_src.reshape(bkg_batch, obj_batch, 3, 1)).reshape(
        bkg_batch, obj_batch, 3
    )
    assert rotmat.shape == (bkg_batch, obj_batch, 3, 3)
    assert shift.shape == (bkg_batch, obj_batch, 3)
    dprint(f"[warp_points_at_random] {rot_angle=}")

    # warp all foreground points
    tgt_points = (
        rotmat.reshape(bkg_batch, obj_batch, 1, 3, 3)
        @ points.reshape(bkg_batch, 1, height * width, 3, 1)
    ).reshape(bkg_batch, obj_batch, height, width, 3) + shift.reshape(
        bkg_batch, obj_batch, 1, 1, 3
    )
    assert tgt_points.shape == (bkg_batch, obj_batch, height, width, 3)

    # ALSO RETURN SE(3) transform
    return (
        tgt_points.reshape(obj_batch, height, width, 3),
        rotmat.reshape(obj_batch, 3, 3),
        shift.reshape(obj_batch, 3),
    )


def detect_collision_above_ground(
    obj_points: Float[torch.Tensor, "n 3"],
    bkg_points: Float[torch.Tensor, "m 3"],
    ground_plane_coeff: Float[torch.Tensor, "4"],
    above_ground_dist_thresh: float = 0.1,
    collision_dist_thresh: float = 0.01,  # less than 1cm can cause false negatives because of the point cloud sparsity
    collision_ratio_thresh: float = 0.01,
) -> bool:
    assert len(obj_points.shape) == 2 and obj_points.shape[1] == 3
    assert len(bkg_points.shape) == 2 and bkg_points.shape[1] == 3
    assert ground_plane_coeff.shape == (4,)
    assert torch.isfinite(obj_points).all()
    assert torch.isfinite(bkg_points).all()
    assert torch.isfinite(ground_plane_coeff).all()
    assert 0 < collision_dist_thresh < above_ground_dist_thresh

    def _get_points_above_ground(
        points: Float[Tensor, "n 3"],
        ground_plane_coeff: Float[Tensor, "4"],
        above_ground_dist_thresh: float,
    ) -> Float[Tensor, "m 3"]:
        points_homogenious = torch.cat([points, torch.ones_like(points[:, 0:1])], dim=-1)
        points_ground_distance = (ground_plane_coeff.reshape(1, 4) * points_homogenious).sum(
            dim=-1
        ).abs() / (1e-6 + torch.norm(ground_plane_coeff[:3], dim=-1))
        return points[points_ground_distance > above_ground_dist_thresh]

    obj_points_above_ground = _get_points_above_ground(
        obj_points, ground_plane_coeff, above_ground_dist_thresh
    )

    if obj_points_above_ground.numel() == 0:
        dprint(
            f"[detect_collision_above_ground] All object points are within {above_ground_dist_thresh}m from the ground."
        )
        return True

    # check knn
    obj_bkg_dist = knn_points(
        obj_points_above_ground.reshape(1, -1, 3),
        bkg_points.reshape(1, -1, 3),
        K=1,
    ).dists
    collision_ratio = (
        torch.sum(obj_bkg_dist < collision_dist_thresh) / obj_points_above_ground.shape[0]
    )
    return collision_ratio > collision_ratio_thresh


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


def compose_objects_with_background(
    obj_imgs: Float[torch.Tensor, "b h w 3"],
    obj_masks: Float[torch.Tensor, "b h w"],
    obj_labels: Int64[torch.Tensor, " b"],
    obj_depths: Float[torch.Tensor, "b h w"],
    bkg_imgs: Float[torch.Tensor, "h w 3"],
    bkg_labels: Int64[torch.Tensor, "h w"],
    bkg_depths: Float[torch.Tensor, "h w"],
    label_offset: int = 1000,
    depth_temperature: float = 0.01,
):
    obj_batch, height, width, channel = obj_imgs.shape
    assert channel == 3
    assert obj_masks.shape == (obj_batch, height, width)
    assert obj_labels.shape == (obj_batch,)
    assert obj_depths.shape == (obj_batch, height, width)
    assert bkg_imgs.shape == (height, width, 3)
    assert bkg_labels.shape == (height, width)
    assert bkg_depths.shape == (height, width)

    # Assuming bkg_labels consists of -1 (undefined) or large id numbers (like COCO)
    # Newly composed objects are assigned labels as <annot_id * label_offset + obj_cnt>
    assert bkg_labels.dtype == torch.int64
    assert (
        obj_batch < bkg_labels[0 <= bkg_labels].min()
    )  # to guarantee the above labeling has no overlap

    # objects should be defined in the finite depth region in the beginning
    assert torch.isfinite(obj_masks).all()
    assert torch.isfinite(obj_depths).all()
    LARGEST_DEPTH = 1e10
    bkg_depths.nan_to_num_(LARGEST_DEPTH).clamp_max_(LARGEST_DEPTH)

    ret_img = bkg_imgs.clone()
    ret_mask = torch.zeros_like(bkg_depths)
    ret_depth = torch.clip(bkg_depths, 0, LARGEST_DEPTH)
    ret_label = bkg_labels * label_offset  # NOTE: label_offset applied
    obj_labels = obj_labels * label_offset  # NOTE: label_offset applied

    for n in range(obj_batch):
        # channel-last
        fg = obj_imgs[n]
        msk = obj_masks[n]
        lbl = obj_labels[n]
        dep = obj_depths[n]

        modal_mask = msk * torch.sigmoid((ret_depth - dep) / depth_temperature)
        ret_img = ret_img - (ret_img - fg) * modal_mask.reshape(height, width, 1)
        ret_mask = torch.max(ret_mask, modal_mask)
        ret_depth = (1 - modal_mask) * ret_depth + modal_mask * dep
        ret_label[modal_mask > 0.5] = (
            lbl + n + 1
        )  # NOTE: 1-indexed to prevent id collision with the original labels

    return ret_img, ret_mask, ret_label


# In[6]:


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


# In[7]:


def process(img_id: int, img_dir: str, coco: COCO, seed: int = 0):
    L.seed_everything(seed)

    # load an image and annotations
    img_path = os.path.join(img_dir, coco.imgs[img_id]["file_name"])
    img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = torch.tensor(img_np, dtype=torch.float32, device=device).permute(2, 0, 1) / 255
    height, width, _ = img_np.shape

    # load annotations
    coco_annot_list = [ann for ann in coco.imgToAnns[img_id] if ann["iscrowd"] == 0]

    with torch.inference_mode():
        # image-related inference
        points, intrinsics = infer_metric_points_and_focal(img)
        semantic_mask = infer_semantic_mask(img)
        gravity_vec = geocalib.calibrate(img)["gravity"].vec3d.reshape(3)  # already L2 normalized

        # extract some useful prediction results
        ground_mask = torch.isin(semantic_mask, oneformer_ground_labels)
        if torch.sum(ground_mask > 0.5) < 1024:
            raise CompositionError("[process] No ground detected in the image")
        sky_mask = torch.isin(semantic_mask, oneformer_sky_labels)
        depth = points[:, :, 2]
        finite_depth_mask = torch.isfinite(depth) * (~sky_mask)

        # refine and reorder masks from large to small (to prevent occlusion)
        obj_size_ann_masks = []
        for ann in coco_annot_list:
            msk_np = coco.annToMask(ann)
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

            if torch.all(mask < 0.5):
                continue

            obj_size_ann_masks.append(((msk_np > 0.5).sum(), ann, mask))
            del msk_np
        obj_size_ann_masks.sort(key=lambda item: item[0], reverse=True)

        # initialize the labelmap and pool unoccluded objects
        unoccluded_annot_ids = []  # NOTE: Not used here, but necessary during json registration
        obj_ann_ids = []
        obj_masks = []
        labelmap = torch.full((height, width), -1, dtype=torch.int64)

        _prev_size = np.inf
        for _size, ann, msk in obj_size_ann_masks:
            assert _prev_size >= _size
            _prev_size = _size
            labelmap[msk > 0.5] = ann["id"]
            if is_object_unoccluded(msk, depth) and is_simply_connected(coco.annToMask(ann)):
                # can put normal weight during training
                unoccluded_annot_ids.append(ann["id"])

                # if, moreover, the bbox is inside the image, it is duplicatable
                bbox = ann["bbox"]  # [sx, sy, w, h]
                bbox_offset = 10
                sx, sy, tx, ty = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                is_bbox_inside = (
                    (bbox_offset <= sx)
                    and (bbox_offset <= sy)
                    and (tx + bbox_offset < width)
                    and (ty + bbox_offset < height)
                )

                if is_bbox_inside and ann["category_id"] in COCO2017_CATEGORIES:
                    obj_ann_ids.append(ann["id"])
                    obj_masks.append(msk)

        if len(obj_ann_ids) == 0:
            raise CompositionError("[process] All objects are partially occluded.")

        obj_batch = len(obj_ann_ids)
        obj_masks = torch.stack(obj_masks, dim=0).reshape(
            obj_batch, height, width
        ) * finite_depth_mask.reshape(1, height, width)

        ##############################

        # denoise object masks as point clouds
        obj_masks = denoise_disconnected_obj_points(
            obj_masks,
            points,
            knn_num=4,
            knn_threshold=0.2,  # VERY SENSITIVE (0.1: a bit too small, 0.3: a bit too large) ####
            remove_isolated_components=True,
            apply_closing=True,
        )

        # select objects that are suitable for warping on the ground
        obj_mask_valid_idx = []
        local_ground_plane_coeff_batch = []
        local_object_plane_coeff_batch = []

        for b, obj_msk in enumerate(obj_masks):
            obj_pts = points[obj_msk > 0.5]

            # 1. Remove objects that are not in contact with the ground
            is_on, local_ground_plane_coeff = is_object_on(obj_pts, points[ground_mask > 0.5])
            if not is_on:
                dprint(f"[process] Object {b} has no near-ground points")
                continue

            # 1.5. Check if the object plane is orthogonal to gravity
            is_plane_horizontal = is_plane_orthogonal_to_gravity(
                local_ground_plane_coeff[:3], gravity_vec
            )
            if not is_plane_horizontal:
                dprint(f"[process] Object {b} plane is not orthogonal to gravity")
                continue

            # 2. Roughly approximatable by a plane
            is_planar, local_object_plane_coeff = is_object_planar(obj_pts)
            if not is_planar:
                dprint(f"[process] Object {b} plane approximation failed")
                continue

            obj_mask_valid_idx.append(b)
            local_ground_plane_coeff_batch.append(local_ground_plane_coeff)
            local_object_plane_coeff_batch.append(local_object_plane_coeff)

        if not obj_mask_valid_idx:
            raise CompositionError("[process] No objects are suitable for warping on the ground")

        obj_batch = len(obj_mask_valid_idx)
        obj_masks = obj_masks[obj_mask_valid_idx].reshape(obj_batch, height, width)
        obj_ann_ids = [obj_ann_ids[i] for i in obj_mask_valid_idx]
        dprint(f"[process] {obj_batch=} (UPDATED)")

        ##############################

        # render objects
        homography_rendered_img = []
        homography_rendered_mask = []
        homography_rendered_depth = []
        homography_rendered_annot_ids = []

        points_for_collision = points[finite_depth_mask].reshape(-1, 3)
        object_sampling_weights = (
            1
            / obj_masks.reshape(obj_batch, height * width)
            .sum(dim=-1)
            .clip(1024, None)
            .float()
            .sqrt()
        )
        dprint(f"[process] {object_sampling_weights=}")

        for k in range(MAX_OBJECT_NUM):
            obj_idx = torch.multinomial(object_sampling_weights, 1)  # k % obj_batch
            obj_msk = obj_masks[obj_idx].reshape(height, width)
            obj_annot_id = obj_ann_ids[obj_idx]

            # warp the object (NOTE: tensor shapes are messy!!! Need to be cleaned up)
            warped_points, rotmat, shift = warp_points_at_random(
                points,
                depth,
                obj_msk.reshape(1, height, width),
                ground_mask,
                local_ground_plane_coeff_batch[obj_idx],
                intrinsics,
                TARGET_DEPTH_REL_RANGE,
            )
            warped_points = warped_points.reshape(height, width, 3)
            warped_obj_pts = warped_points[obj_msk > 0.5]

            # collision check
            if detect_collision_above_ground(
                warped_obj_pts,
                points_for_collision,
                local_ground_plane_coeff_batch[obj_idx],
                above_ground_dist_thresh=ABOVE_GROUND_DIST_THRESH,
                collision_dist_thresh=COLLISION_DIST_THRESH,
                collision_ratio_thresh=COLLISION_RATIO_THRESH,
            ):
                dprint(f"[process] Collision detected with object {k}")
                continue
            else:
                points_for_collision = torch.cat([points_for_collision, warped_obj_pts], dim=0)
                assert (
                    len(points_for_collision.shape) == 2 and points_for_collision.shape[1] == 3
                ), f"{points_for_collision.shape=}"

            # approximate warping with a homography by tracing the four bbox corners
            sx, sy, tx, ty = torchvision.ops.masks_to_boxes(
                obj_msk.reshape(1, height, width) > 0.5
            ).reshape(4)
            bbox_pixel_corners = torch.tensor(
                [[sx, sy], [tx, sy], [sx, ty], [tx, ty]], device=sx.device
            )
            fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
            bbox_world_corners = torch.stack(
                [
                    (bbox_pixel_corners[:, 0] - cx) / fx,
                    (bbox_pixel_corners[:, 1] - cy) / fy,
                    torch.ones_like(bbox_pixel_corners[:, 0]),
                ],
                dim=-1,
            )

            obj_pa, obj_pb, obj_pc, obj_pd = local_object_plane_coeff_batch[obj_idx]
            bbox_corners_plane_depth = -obj_pd / (
                obj_pa * bbox_world_corners[:, 0] + obj_pb * bbox_world_corners[:, 1] + obj_pc
            )
            assert torch.all(bbox_corners_plane_depth > 0), f"{bbox_corners_plane_depth=}"
            bbox_corners_unproj = bbox_world_corners * bbox_corners_plane_depth.reshape(4, 1)
            bbox_corners_warped = rotmat.reshape(1, 3, 3) @ bbox_corners_unproj.reshape(
                4, 3, 1
            ) + shift.reshape(1, 3, 1)
            bbox_corners_warped_projected = (intrinsics @ bbox_corners_warped).reshape(4, 3)
            bbox_pixel_corners_tgt = (
                bbox_corners_warped_projected[:, :2] / bbox_corners_warped_projected[:, 2:3]
            )

            # render with homography
            homographies = find_homography_dlt(
                bbox_pixel_corners.float().reshape(1, 4, 2),
                bbox_pixel_corners_tgt.float().reshape(1, 4, 2),
            )
            image_mask_depth = torch.cat(
                [
                    img.reshape(1, 3, height, width),
                    obj_msk.reshape(1, 1, height, width),
                    warped_points[:, :, 2].reshape(1, 1, height, width),
                ],
                dim=1,
            )
            image_mask_depth_warped = warp_perspective(
                image_mask_depth,
                homographies,
                (height, width),
                mode="bilinear",
                align_corners=False,
            )

            # save results
            homography_rendered_img.append(
                image_mask_depth_warped[:, :3, :, :].permute(0, 2, 3, 1)
            )
            homography_rendered_mask.append(image_mask_depth_warped[:, 3, :, :])
            homography_rendered_depth.append(image_mask_depth_warped[:, 4, :, :])
            homography_rendered_annot_ids.append(obj_annot_id)

        # compose the rendered images
        if len(homography_rendered_img) == 0:
            raise CompositionError("[process] All warped objects collide the scene")

        obj_batch = len(homography_rendered_img)
        dprint(f"[process] {obj_batch=} (UPDATED)")
        homography_rendered_img = torch.cat(homography_rendered_img, dim=0).reshape(
            obj_batch, height, width, 3
        )
        homography_rendered_mask = torch.cat(homography_rendered_mask, dim=0).reshape(
            obj_batch, height, width
        )
        homography_rendered_depth = torch.cat(homography_rendered_depth, dim=0).reshape(
            obj_batch, height, width
        )
        homography_rendered_annot_ids = torch.tensor(
            homography_rendered_annot_ids, dtype=torch.int64, device=device
        ).reshape(obj_batch)

        # color augmentation
        homography_rendered_img = change_object_color_hsv(
            homography_rendered_img, homography_rendered_mask, sv_max_shift=0.2
        )

        composed_img, composed_mask, composed_label = compose_objects_with_background(
            homography_rendered_img,
            homography_rendered_mask,
            homography_rendered_annot_ids,
            homography_rendered_depth,
            img.permute(1, 2, 0),
            labelmap,
            depth,
            label_offset=LABEL_OFFSET,
        )
        return (
            composed_img,
            composed_mask,
            composed_label,
            homography_rendered_img,
            homography_rendered_mask,
            homography_rendered_depth,
            depth,
            unoccluded_annot_ids,
        )


# In[8]:


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


def is_enough_visible(
    visible_mask: Float[torch.Tensor, "h w"],
    visible_area: float,
    amodal_area: float,
    visible_area_ratio_thresh: float = 0.05,
    border_area_ratio_thresh: float = 0.25,
):
    # False if more than 95% are occluded
    if visible_area < amodal_area * visible_area_ratio_thresh:
        return False

    # False if located around the image border
    border_ratio = 0.01
    height, width = visible_mask.shape
    height_border = int(height * border_ratio)
    width_border = int(width * border_ratio)
    visible_center_area = visible_mask[
        height_border:-height_border, width_border:-width_border
    ].sum()
    if visible_center_area < visible_area * border_area_ratio_thresh:
        return False

    return True


def register_composed_result(
    coco: COCO,
    img_id: int,
    composed_img: Float[Tensor, "h w 3"],
    composed_mask: Float[Tensor, "h w"],
    composed_label: Int64[Tensor, "h w"],
    rendered_img: Float[Tensor, "b h w 3"],
    rendered_mask: Float[Tensor, "b h w"],
    rendered_depth: Float[Tensor, "b h w"],
    unoccluded_annot_ids: list[int],
    label_offset: int = 1000,
) -> tuple[
    list[dict[str, Any]], dict[int, Optional[np.ndarray]], dict[int, Optional[np.ndarray]], int
]:
    obj_batch, height, width, channel = rendered_img.shape
    assert channel == 3
    assert rendered_mask.shape == (obj_batch, height, width)
    assert composed_img.shape == (height, width, 3)
    assert composed_mask.shape == (height, width)
    assert composed_label.shape == (height, width)

    annotations = {ann["id"]: ann for ann in coco.imgToAnns[img_id]}
    all_annot_ids = annotations.keys()
    new_annot_list_per_img = []
    new_rgb_dict_per_img = {}
    new_depth_dict_per_img = {}
    composed_obj_num = 0

    for lbl_id in torch.unique(composed_label):
        lbl_id = lbl_id.item()
        # lbl_id = -1 is undefined
        if lbl_id < 0:
            continue

        ann_id, composite_id = divmod(lbl_id, label_offset)
        composite_id -= 1  # NOTE: the composition 'residual' labels are 1-indexed, so we need to decrement here
        assert ann_id in all_annot_ids, f"{torch.unique(composed_label)=}"
        ann = annotations[ann_id]
        msk = composed_label == lbl_id

        # existing mask
        if composite_id < 0:  # NOTE: composite_id is already decremented
            visible_segm, visible_bbox, visible_area = get_contour_bbox_area(msk)
            amodal_segm, amodal_bbox, amodal_area = ann["segmentation"], ann["bbox"], ann["area"]
            mask_type = "real_whole" if (ann_id in unoccluded_annot_ids) else "real_partial"
            if not is_enough_visible(
                msk,
                visible_area,
                amodal_area,
                visible_area_ratio_thresh=0.05,
                border_area_ratio_thresh=0.0,
            ):
                continue
            amodal_rgb = None
            amodal_depth = None

        # new mask
        else:
            visible_segm, visible_bbox, visible_area = get_contour_bbox_area(msk)
            amodal_segm, amodal_bbox, amodal_area = get_contour_bbox_area(
                rendered_mask[composite_id]
            )
            mask_type = "syn"
            if not is_enough_visible(
                msk,
                visible_area,
                amodal_area,
                visible_area_ratio_thresh=0.05,
                border_area_ratio_thresh=0.25,
            ):
                continue
            sx, sy, w, h = amodal_bbox
            sx, sy, tx, ty = int(sx), int(sy), int(sx + w), int(sy + h)
            amodal_rgb = rendered_img[composite_id, sy:ty, sx:tx].cpu().numpy()
            amodal_depth = rendered_depth[composite_id, sy:ty, sx:tx].cpu().numpy()
            assert amodal_rgb.shape == (h, w, 3), f"{amodal_rgb.shape=}, {amodal_bbox=}"
            assert amodal_depth.shape == (h, w), f"{amodal_depth.shape=}, {amodal_bbox=}"
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
        new_rgb_dict_per_img[lbl_id] = amodal_rgb
        new_depth_dict_per_img[lbl_id] = amodal_depth

    return new_annot_list_per_img, new_rgb_dict_per_img, new_depth_dict_per_img, composed_obj_num


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


def save_depth(depth_outpath: str, depth: Float[np.ndarray | Tensor, "h w"]):
    assert depth_outpath.endswith(".npy")
    assert len(depth.shape) == 2
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    np.save(depth_outpath, depth.astype(np.float16))


if __name__ == "__main__":
    coco_root_dir = "/media/ryotaro/ssd1/coco/"
    is_train_or_val = "train"
    image_save_dir = os.path.join(
        coco_root_dir, f"{is_train_or_val}2017_composed_with_depth/image"
    )
    depth_save_dir = os.path.join(
        coco_root_dir, f"{is_train_or_val}2017_composed_with_depth/depth"
    )

    img_dir = os.path.join(coco_root_dir, f"{is_train_or_val}2017")
    coco = COCO(os.path.join(coco_root_dir, f"annotations/instances_{is_train_or_val}2017.json"))
    annot_id_list = coco.getAnnIds(catIds=[])

    new_img_list = []
    new_img_set = set()
    new_annot_list = []

    # load json if exists (assuming the process was interrupted in the middle)
    json_outpath = os.path.join(
        coco_root_dir,
        f"annotations/instances_{is_train_or_val}2017_kakuda_composition_labels_with_depth.json",
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
    os.makedirs(depth_save_dir, exist_ok=True)

    for img_id in tqdm(coco.getImgIds()):
        if img_id in new_img_set:
            continue

        # compose
        for trial in range(COMPOSITION_RETRY_NUM):
            new_annots_per_img = []

            try:
                (
                    composed_img,
                    composed_mask,
                    composed_label,
                    homography_rendered_img,
                    homography_rendered_mask,
                    homography_rendered_depth,
                    depth,
                    unoccluded_annot_ids,
                ) = process(img_id, img_dir, coco, seed=trial)
            except CompositionError as e:
                print(f"[main] {img_id}: {e}")
                new_annots_per_img = []
                break
            except Exception as e:
                raise Exception(img_id, e)

            # register
            new_annots_per_img, new_rgbs_per_img, new_depths_per_img, composed_obj_num = (
                register_composed_result(
                    coco,
                    img_id,
                    composed_img,
                    composed_mask,
                    composed_label,
                    homography_rendered_img,
                    homography_rendered_mask,
                    homography_rendered_depth,
                    unoccluded_annot_ids,
                    label_offset=LABEL_OFFSET,
                )
            )
            # if no objects are composed <even though the ground is visible>, discard the image
            if composed_obj_num == 0:
                dprint(
                    f"\n[main] {img_id}: Composed objects are all invisible (Trial: {trial + 1}/{COMPOSITION_RETRY_NUM}).\n"
                )
                new_annots_per_img = []
                continue
            else:
                break

        if new_annots_per_img:
            new_annot_list.extend(new_annots_per_img)

            # save the background image (NOTE: amodal texture is so far not saved!!!)
            image_save_path = os.path.join(image_save_dir, coco.imgs[img_id]["file_name"])
            ret_uint8 = np.clip(255 * composed_img.cpu().numpy(), 0, 255).astype(np.uint8)
            cv2.imwrite(image_save_path, ret_uint8[:, :, ::-1])
            new_img_list.append(coco.imgs[img_id])
            new_img_set.add(img_id)
            dprint(f"\n{img_id}: saved! (Composed object num: {composed_obj_num})\n")

            # save the background depth
            depth_save_path = os.path.join(
                depth_save_dir,
                coco.imgs[img_id]["file_name"].replace(".jpg", ".npy").replace(".png", ".npy"),
            )
            save_depth(depth_save_path, depth)

            # save the object image
            for annot_id, rgb in new_rgbs_per_img.items():
                if rgb is not None:
                    cv2.imwrite(
                        os.path.join(image_save_dir, f"{annot_id}.jpg"),
                        np.clip(255 * rgb, 0, 255).astype(np.uint8)[:, :, ::-1],
                    )
            for annot_id, dep in new_depths_per_img.items():
                if dep is not None:
                    save_depth(os.path.join(depth_save_dir, f"{annot_id}.npy"), dep)

        if new_img_list and len(new_img_list) % JSON_SAVE_FREQ == 0:
            save_json(new_img_list, new_annot_list, json_outpath, coco)

    # save into a json file
    save_json(new_img_list, new_annot_list, json_outpath, coco)
    print(f"Finished! {json_outpath=}")
