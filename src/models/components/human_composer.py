import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from kornia.geometry.homography import find_homography_dlt
from kornia.geometry.transform import warp_perspective
from pytorch3d.transforms import axis_angle_to_matrix
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

from geocalib import GeoCalib
from third_party.MaGGIe.maggie.network.arch import MaGGIe
from third_party.MoGe.moge.model import MoGeModel
from third_party.PCTNet.iharm.inference.utils import load_model
from third_party.StyleGAN_Human import dnnlib, legacy

STYLEGAN_HUMAN_PKL = (
    "/home/ryotaro/github/StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl"
)
HARMONIZER_WEIGHT = (
    "/home/ryotaro/my_works/HoGe/third_party/PCTNet/pretrained_models/PCTNet_ViT.pth"
)

LARGEST_DEPTH = 1000


class HumanCompositionError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


# >>> Plane Fitting >>>


def PCA(data, correlation: bool = False, sort: bool = True):
    """Applies Batch PCA to input tensor.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of shape (B, N, M)
        B: batch size, N: number of records, M: number of features

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
    # Subtract mean along record dimension
    mean = data.mean(dim=1, keepdim=True)
    data_adjusted = data - mean

    # Compute matrix based on correlation or covariance
    if correlation:
        # Compute correlation for each batch
        matrix = torch.stack([torch.corrcoef(batch_data.T) for batch_data in data_adjusted])
    else:
        # https://stackoverflow.com/questions/71357619/how-do-i-compute-batched-sample-covariance-in-pytorch
        def batch_cov(points):
            B, N, D = points.size()
            mean = points.mean(dim=1).unsqueeze(1)
            diffs = (points - mean).reshape(B * N, D)
            prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
            bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
            return bcov  # (B, D, D)

        matrix = batch_cov(data_adjusted)

    # Compute eigenvalues and eigenvectors for each batch matrix
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    except torch._C._LinAlgError as e:
        raise HumanCompositionError(f"[PCA] {e}")

    # Sort if required (in descending order)
    if sort:
        # Flip eigenvalues and eigenvectors to descending order
        sorted_indices = eigenvalues.argsort(dim=1, descending=True)
        batch_indices = torch.arange(eigenvalues.size(0))[:, None]

        eigenvalues = eigenvalues[batch_indices, sorted_indices]
        eigenvectors = eigenvectors[batch_indices, :, sorted_indices]

    return eigenvalues, eigenvectors.permute(0, 2, 1)


def best_fitting_plane(points, equation: bool = False):
    """Computes the best fitting plane for batched points.

    Parameters
    ----------
    points : torch.Tensor
        Input tensor of shape (B, N, 3)
        B: batch size, N: number of points, 3: x,y,z coordinates

    equation : bool, optional
        If True, return plane coefficients.
        If False, return point and normal vector.

    Returns
    -------
    If equation=False:
        point : torch.Tensor of shape (B, 3)
        normal : torch.Tensor of shape (B, 3)

    If equation=True:
        a, b, c, d : torch.Tensor of shape (B,)
    """
    # Compute PCA for each batch of points
    eigenvalues, eigenvectors = PCA(points)

    # The normal is the last eigenvector (smallest eigenvalue)
    normal = eigenvectors[:, :, 2]

    # Get mean point for each batch
    point = points.mean(dim=1)

    if equation:
        # Compute plane equation coefficients
        a, b, c = normal.T
        d = -(normal * point).sum(dim=1)
        return a, b, c, d
    else:
        return point, normal


# <<< Plane Fitting <<<


class HumanComposer(nn.Module):
    def __init__(
        self,
        max_human_num_range: tuple[int, int] = (6, 48),
        human_min_depth_range: tuple[int, int] = (0.5, 5),
        human_max_depth_range: tuple[int, int] = (15, 20),
        verbose: bool = False,
    ):
        super().__init__()
        assert (
            len(max_human_num_range) == 2 and 0 < max_human_num_range[0] < max_human_num_range[1]
        )
        assert (
            len(human_min_depth_range) == 2 and human_min_depth_range[0] < human_min_depth_range[1]
        )
        assert (
            len(human_max_depth_range) == 2 and human_max_depth_range[0] < human_max_depth_range[1]
        )
        assert human_min_depth_range[1] < human_max_depth_range[0]
        self.max_human_num_range = max_human_num_range
        self.human_min_depth_range = human_min_depth_range
        self.human_max_depth_range = human_max_depth_range
        self.verbose = verbose

        # StyleGAN-Human
        with dnnlib.util.open_url(STYLEGAN_HUMAN_PKL) as f:
            self.stylegan_human = legacy.load_network_pkl(f)["G_ema"]

        # MaGGIe Human Matting
        self.matting_model = MaGGIe.from_pretrained("chuonghm/maggie-image-him50k-cvpr24").cpu()

        # OneFormer
        self.oneformer_processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_dinat_large"
        )
        self.oneformer_task_inputs = self.oneformer_processor._preprocess_text(
            ["the task is semantic"]
        )
        self.oneformer = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_dinat_large"
        )

        # Metric depth
        self.geocalib = GeoCalib()
        self.metric3dv2 = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
        self.moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl")

        # Harmonizer
        model_type = "ViT_pct"
        self.harmonizer = load_model(model_type, HARMONIZER_WEIGHT, verbose=False)

        # start from eval mode
        self.eval()

    def run_oneformer(self, img: Float[torch.Tensor, "b 3 h w"]) -> Float[torch.Tensor, "b 1 h w"]:
        b, c, h_ori, w_ori = img.shape
        assert c == 3

        mean = torch.tensor(
            self.oneformer_processor.image_processor.image_mean,
            dtype=img.dtype,
            device=img.device,
        ).reshape(1, 3, 1, 1)

        std = torch.tensor(
            self.oneformer_processor.image_processor.image_std,
            dtype=img.dtype,
            device=img.device,
        ).reshape(1, 3, 1, 1)

        scale_factor = self.oneformer_processor.image_processor.size["shortest_edge"] / min(
            img.shape[-2:]
        )
        pixel_values = F.interpolate(img, scale_factor=scale_factor, mode="bilinear")
        pixel_values *= 255 * self.oneformer_processor.image_processor.rescale_factor
        pixel_values = (pixel_values - mean) / std

        b, c, h, w = pixel_values.shape
        semantic_inputs = {
            "pixel_values": pixel_values,
            "pixel_mask": torch.ones(b, h, w, dtype=torch.int64, device=img.device),
            "task_inputs": self.oneformer_task_inputs.repeat(b, 1).to(img.device),
        }
        semantic_outputs = self.oneformer(**semantic_inputs)
        predicted_semantic_map = self.oneformer_processor.post_process_semantic_segmentation(
            semantic_outputs,
            target_sizes=[(h_ori, w_ori)] * b,
        )
        predicted_semantic_map = torch.stack(predicted_semantic_map, dim=0).unsqueeze(
            1
        )  # (b, 1, h_ori, w_ori)

        return predicted_semantic_map

    def segment_ground(
        self, img: Float[torch.Tensor, "b 3 h w"]
    ) -> Float[torch.Tensor, "b 1 h w"]:
        predicted_semantic_map = self.run_oneformer(img)
        # In ADE20K
        labels = torch.tensor(
            [
                3,  # 3 => floor
                6,  # 6 => road, route
                9,  # 9 => grass
                11,  # 11 => sidewalk, pavement
                13,  # 13 => earth, ground
                28,  # 28 => rug
                53,  # 53 => stairs
                59,  # 59 => stairway, staircase
                96,  # 96 => escalator, moving staircase, moving stairway
                121,  # 121 => step, stair
            ],
            dtype=predicted_semantic_map.dtype,
            device=predicted_semantic_map.device,
        )
        ground_mask = torch.isin(predicted_semantic_map, labels)
        return ground_mask.float()

    def generate_human(
        self,
        batch_size: int,
        truncation_psi: float = 1.0,
        noise_mode: str = "const",
        device: str = "cuda",
    ) -> Float[torch.Tensor, "batch_size 3 1024 512"]:
        assert noise_mode in ["const", "random", "none"]
        label = torch.zeros([batch_size, self.stylegan_human.c_dim], device=device)
        z = torch.randn(batch_size, self.stylegan_human.z_dim, dtype=torch.float32, device=device)
        w = self.stylegan_human.mapping(z, label, truncation_psi=truncation_psi)
        human = self.stylegan_human.synthesis(w, noise_mode=noise_mode, force_fp32=True)
        human = torch.clip((human + 1) / 2, 0, 1)  # (-1, 1) -> (0, 1)
        return human

    def segment_human(
        self, human: Float[torch.Tensor, "b 3 h w"]
    ) -> Float[torch.Tensor, "b 1 h w"]:
        b, c, ori_h, ori_w = human.shape
        assert c == 3

        predicted_semantic_map = self.run_oneformer(human)
        # In ADE20K
        human_mask = (predicted_semantic_map == 12).float()  # 12 => person

        def _resize(
            img: Float[torch.Tensor, "b c h w"],
            size: int,
            to_short_size: bool,
            multiple_of: int = 1,
            padding_val: float = 1,
        ):
            assert size > 0
            _, _, ori_h, ori_w = img.shape
            ratio = size / min(ori_h, ori_w) if to_short_size else size / max(ori_h, ori_w)
            h, w = int(ori_h * ratio), int(ori_w * ratio)
            h_pad, w_pad = (multiple_of - h % multiple_of) % multiple_of, (
                multiple_of - w % multiple_of
            ) % multiple_of
            # print(f"{ori_h=}, {ori_w=}, {h=}, {w=}, {h_pad=}, {w_pad=}")
            if h_pad > 0 or w_pad > 0:
                ret = torch.full(
                    (b, c, h + h_pad, w + w_pad),
                    fill_value=padding_val,
                    dtype=human.dtype,
                    device=human.device,
                )
                ret[:, :, :h, :w] = F.interpolate(img, (h, w), mode="bilinear")
            else:
                ret = F.interpolate(img, (h, w), mode="bilinear")
            return ret, h, w

        # preprocess matting image
        human_resized, h, w = _resize(
            human, 576, to_short_size=True, multiple_of=64, padding_val=1
        )
        segmask_resized, _, _ = _resize(
            human_mask, 576, to_short_size=True, multiple_of=64, padding_val=0
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=human.dtype, device=human.device).reshape(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], dtype=human.dtype, device=human.device).reshape(
            1, 3, 1, 1
        )
        human_resized = (human_resized - mean) / std

        # error if segmask is empty
        if not torch.any(segmask_resized > 0.5) or not torch.isfinite(segmask_resized).all():
            raise HumanCompositionError(
                f"Erroneous segmask: {(segmask_resized > 0.5).flatten(1).sum(dim=-1)=}"
            )

        # matting
        batch = {"image": human_resized.unsqueeze(1), "mask": segmask_resized.unsqueeze(1)}
        try:
            output = self.matting_model(batch)
        except ValueError as e:
            print(e)
            print(f"{(segmask_resized > 0.5).flatten(1).sum(dim=-1)=}")
            raise HumanCompositionError(e)

        # Postprocess alpha matte
        alpha = output["refined_masks"].squeeze(1)
        alpha = F.interpolate(alpha[..., :h, :w], (ori_h, ori_w), mode="bilinear")
        alpha[alpha <= 1.0 / 255.0] = 0.0
        alpha[alpha >= 254.0 / 255.0] = 1.0

        return alpha

    def get_metric_depth(
        self,
        img: Float[torch.Tensor, "b 3 h w"],
        cam_focal_len: Float[torch.Tensor, " b"] = None,
    ):
        batch, channel, height, width = img.shape
        assert channel == 3

        # calibration
        if cam_focal_len is None:
            cam_focal_len = []
            for b in range(batch):
                calib_result = self.geocalib.calibrate(img[b])
                cam_focal_len.append(calib_result["camera"].f.mean())
            cam_focal_len = torch.stack(cam_focal_len)

        # metric3d
        input_size = (616, 1064)  # for vit model
        # input_size = (544, 1216) # for convnext model
        scale = min(input_size[0] / height, input_size[1] / width)
        img_resized = F.interpolate(img, scale_factor=scale, mode="bilinear")
        # padding to input_size
        img_padded = torch.empty((batch, channel, *input_size), dtype=img.dtype, device=img.device)
        img_padded[:, 0, :, :] = 123.675 / 255
        img_padded[:, 1, :, :] = 116.28 / 255
        img_padded[:, 2, :, :] = 103.53 / 255
        _, _, h, w = img_resized.shape
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        assert pad_h == 0 or pad_w == 0
        if pad_h == 0 and pad_w > 0:
            img_padded[:, :, :, pad_w_half : -(pad_w - pad_w_half)] = img_resized
        elif pad_h > 0 and pad_w == 0:
            img_padded[:, :, pad_h_half : -(pad_h - pad_h_half), :] = img_resized
        elif pad_h == pad_w == 0:
            img_padded = img_resized
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # normalize
        mean = (
            torch.tensor([123.675, 116.28, 103.53], dtype=img.dtype, device=img.device).reshape(
                1, 3, 1, 1
            )
            / 255
        )
        std = (
            torch.tensor([58.395, 57.12, 57.375], dtype=img.dtype, device=img.device).reshape(
                1, 3, 1, 1
            )
            / 255
        )
        img_padded = torch.div((img_padded - mean), std)

        # >>>>>>>>>>>>>>>>>>>> canonical camera space >>>>>>>>>>>>>>>>>>>>
        # inference
        pred_depth, confidence, output_dict = self.metric3dv2.inference({"input": img_padded})

        # un pad
        pred_depth = pred_depth[
            :,
            :,
            pad_info[0] : input_size[0] - pad_info[1],
            pad_info[2] : input_size[1] - pad_info[3],
        ]

        # upsample to original size
        pred_depth = F.interpolate(pred_depth, (height, width), mode="bilinear")
        # >>>>>>>>>>>>>>>>>>>> canonical camera space >>>>>>>>>>>>>>>>>>>>

        # de-canonical transform
        canonical_to_real_scale = (
            cam_focal_len * scale / 1000.0
        )  # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale.reshape(
            batch, 1, 1, 1
        )  # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)

        return pred_depth.squeeze(1), cam_focal_len  # (b, h, w): align with the DepthPro output

    @staticmethod
    def get_human_foot_pixel_coord_and_height(
        human_mask: Float[torch.Tensor, "b 1 h w"],
        index: str = "ij",
    ):
        batch, channel, height, width = human_mask.shape
        assert channel == 1

        human_mask_vertical = torch.any(human_mask > 0.5, dim=-1)
        human_mask_horizontal = torch.any(human_mask > 0.5, axis=-2)

        mask_vertical_idx, _, mask_vertical_px = human_mask_vertical.nonzero(as_tuple=True)
        foot_pos_in_fg_y = mask_vertical_px[
            mask_vertical_idx.bincount().cumsum(dim=0) - 1
        ]  # lowest pixel

        mask_horizontal_idx, _, mask_horizontal_px = human_mask_horizontal.nonzero(as_tuple=True)
        foot_pos_in_fg_x = torch.bincount(
            mask_horizontal_idx, weights=mask_horizontal_px
        ) / torch.bincount(mask_horizontal_idx)

        foot_pos_in_fg = torch.stack([foot_pos_in_fg_y, foot_pos_in_fg_x], dim=-1)
        current_human_pixel_height = torch.sum(
            human_mask_vertical.reshape(batch, height), dim=1
        )  # NOTE: Should be more sophisticated
        assert foot_pos_in_fg.shape == (batch, 2)

        if index == "ij":
            pass
        elif index == "xy":
            foot_pos_in_fg = torch.flip(foot_pos_in_fg, dims=(1,))
        else:
            raise ValueError(f"Invalid {index=}")

        return foot_pos_in_fg.to(human_mask.dtype), current_human_pixel_height

    def place_humans_in_the_scene(
        self,
        humans: Float[torch.Tensor, "b_human 3 h w"],
        humans_mask: Float[torch.Tensor, "b_human 1 h w"],
        bkg_metric_depth: Float[torch.Tensor, "b h_bkg w_bkg"],
        plane_coeff: Float[torch.Tensor, "b 4"],
        intrinsics: Float[torch.Tensor, "b 3 3"],
    ) -> tuple[Float[torch.Tensor, "b b_human 4 h w"], Float[torch.Tensor, "b b_human 1 h w"]]:
        """All the humans are pasted in each of the background scene."""
        bkg_batch, bkg_height, bkg_width = bkg_metric_depth.shape
        human_batch, _, human_height, human_width = humans.shape

        PA, PB, PC, PD = torch.split(plane_coeff, 1, dim=1)
        assert PA.shape == (bkg_batch, 1), f"{PA.shape=}"
        ground_normal = plane_coeff[:, :3]
        bkg_max_depth = torch.max(bkg_metric_depth.reshape(bkg_batch, -1), dim=1).values

        foot_pos_in_fg, current_human_pixel_height = (
            __class__.get_human_foot_pixel_coord_and_height(humans_mask, index="xy")
        )

        humans_actual_height = torch.normal(
            1.7, 0.07, (human_batch,), dtype=humans.dtype, device=humans.device
        )
        humans_max_depth = torch.empty_like(bkg_max_depth).uniform_(
            self.human_max_depth_range[0], self.human_max_depth_range[1]
        )
        humans_min_depth = self.human_min_depth_range[0] + torch.rand_like(bkg_max_depth) * (
            bkg_max_depth.clamp(self.human_min_depth_range[0], self.human_min_depth_range[1])
            - self.human_min_depth_range[0]
        )
        humans_foot_depth = humans_min_depth + torch.rand(
            (bkg_batch, human_batch), dtype=humans.dtype, device=humans.device
        ) * (humans_max_depth - humans_min_depth).reshape(
            bkg_batch, 1
        )  # (bkg_batch, human_batch)
        assert humans_foot_depth.shape == (bkg_batch, human_batch)
        if self.verbose:
            print(f"[place_humans_in_the_scene] {humans_min_depth=}")
            print(f"[place_humans_in_the_scene] {humans_max_depth=}")
            print(f"[place_humans_in_the_scene] {humans_actual_height=}")
            print(f"[place_humans_in_the_scene] {humans_foot_depth=}")

        """Put humans on the intersection line of ax+by+cz+d=0 (plane_eq) and z=humans_foot_depth.
        Note that nearly a=c=0 in most cases.

        The intersection line is parametrized by (x, (-d - c * humans_foot_depth - ax) / b, humans_foot_depth)
        The free variable x is constrained by the camera FoV.
        """
        # foot position in 3D space
        foot_pos_in_bg_x_max = (
            (bkg_width / 2) * humans_foot_depth / intrinsics[:, 0, 0].reshape(bkg_batch, 1)
        )
        foot_pos_in_bg_x = (torch.rand_like(humans_foot_depth) * 2 - 1) * foot_pos_in_bg_x_max
        foot_pos_in_bg_y = (-PD - PC * humans_foot_depth - PA * foot_pos_in_bg_x) / PB
        foot_pos_in_bg = torch.stack(
            [foot_pos_in_bg_x, foot_pos_in_bg_y, humans_foot_depth], dim=-1
        )
        assert foot_pos_in_bg.shape == (bkg_batch, human_batch, 3), f"{foot_pos_in_bg.shape=}"

        # define the corner points of the human plane in the 3D space, PARPENDICULAR TO THE PRINCIPAL AXIS for ease
        # (NOTE: metric depth is correct in all directions x, y, and z)
        scale_factor = humans_actual_height / current_human_pixel_height
        corner_pixels = torch.tensor(
            [[[0, 0], [human_width, 0], [0, human_height], [human_width, human_height]]],
            dtype=humans.dtype,
            device=humans.device,
        ).repeat(
            human_batch, 1, 1
        )  # (b_human, 4, 2)

        corner_points_plane_xy = (
            corner_pixels - foot_pos_in_fg.reshape(human_batch, 1, 2)
        ) * scale_factor.reshape(human_batch, 1, 1)
        corner_points_plane_z = torch.zeros_like(corner_points_plane_xy[..., 0:1])
        corner_points_foot_origin = torch.cat(
            [corner_points_plane_xy, corner_points_plane_z], dim=-1
        )
        assert corner_points_foot_origin.shape == (
            human_batch,
            4,
            3,
        ), f"{corner_points_foot_origin.shape=}"

        # rotate the human planes so the body axis aligns with the ground normal with the foot point fixed
        initial_normal = torch.zeros((bkg_batch, 3), dtype=humans.dtype, device=humans.device)
        initial_normal[:, 1] = torch.sign(ground_normal[:, 1])
        target_normal = F.normalize(ground_normal, dim=1)
        rotation_axis = F.normalize(torch.linalg.cross(initial_normal, target_normal), dim=1)
        angle = torch.acos(torch.sum(initial_normal * target_normal, dim=1, keepdim=True))
        rotmat = axis_angle_to_matrix(angle * rotation_axis)  # (bkg_batch, 3, 3)

        corner_points = (
            rotmat.reshape(bkg_batch, 1, 3, 3)
            @ corner_points_foot_origin.reshape(1, human_batch * 4, 3, 1)
        ).reshape(bkg_batch, human_batch, 4, 3) + foot_pos_in_bg.reshape(
            bkg_batch, human_batch, 1, 3
        )
        assert corner_points.shape == (bkg_batch, human_batch, 4, 3)

        # get plane depth maps (bilinearly interpolate the four corner depth values)
        humans_plane_depth = F.interpolate(
            corner_points[..., 2].reshape(-1, 1, 2, 2),
            (human_height, human_width),
            mode="bilinear",
            align_corners=True,
        ).reshape(bkg_batch, human_batch, 1, human_height, human_width)

        # project these four points to the image plane
        corner_points_projected = torch.matmul(
            intrinsics, corner_points.reshape(bkg_batch, human_batch * 4, 3, 1)
        ).reshape(bkg_batch, human_batch, 4, 3)
        corner_points_projected_xy = (
            corner_points_projected[..., :2] / corner_points_projected[..., 2:3]
        )

        # get homography and warp the entire human images/masks to the image plane
        homographies = find_homography_dlt(
            corner_pixels.reshape(human_batch, 4, 2),
            corner_points_projected_xy.reshape(bkg_batch * human_batch, 4, 2),
        )
        humans_height_map = (
            foot_pos_in_fg[:, 1].reshape(human_batch, 1)
            - torch.linspace(0, human_height - 1, human_height, device=humans.device)
            .reshape(1, -1)
            .expand(human_batch, -1)
        ) * scale_factor.reshape(human_batch, 1)
        humans_height_map = humans_height_map.reshape(human_batch, 1, human_height, 1).expand(
            -1, -1, -1, human_width
        )  # NEW!!! FOR COLLISION DETECTION WITH THE BACKGROUND
        humans_image_mask_depth = torch.cat(
            [
                humans[None].expand(bkg_batch, -1, -1, -1, -1),
                humans_mask[None].expand(bkg_batch, -1, -1, -1, -1),
                humans_plane_depth,
                humans_height_map[None],
            ],
            dim=2,
        )
        try:
            humans_image_mask_depth_warped = warp_perspective(
                humans_image_mask_depth.reshape(
                    bkg_batch * human_batch, 6, human_height, human_width
                ),
                homographies,
                (bkg_height, bkg_width),
            )
            humans_image_mask_depth_warped = humans_image_mask_depth_warped.reshape(
                bkg_batch, human_batch, 6, bkg_height, bkg_width
            )
        except torch._C._LinAlgError as e:
            print(f"[place_human_in_the_scene] warp_perspective: {e}")
            raise HumanCompositionError(e)
        humans_image_mask_warped, humans_depth_warped, humans_height_warped = torch.split(
            humans_image_mask_depth_warped, [4, 1, 1], dim=2
        )

        # detect collision (colliding objects are moved to LARGEST_DEPTH)
        def _judge_collision(
            humans_image_mask,
            humans_depth,
            humans_height,
            bkg_metric_depth,
            ABOVE_GROUND_THRESH=0.3,
            COLLISION_DIST_THRESH=0.001,
            COLLISION_JUDGE_THRESH=10,  # NOTE: pixel num
        ):

            # Collision detection between humans and the scene (only the depth difference is fine to check)
            humans_bkg_depth_diff = torch.abs(
                humans_depth - rearrange(bkg_metric_depth, "b h w -> b () () h w")
            )  # (bkg_batch, obj_batch, 1, h, w)
            humans_mask_binary = humans_image_mask[:, :, 3:4, :, :] < 0.99
            humans_bkg_depth_diff_mask = humans_mask_binary * (humans_height > ABOVE_GROUND_THRESH)
            humans_bkg_depth_diff[humans_bkg_depth_diff_mask] = LARGEST_DEPTH
            humans_bkg_depth_diff.nan_to_num_(LARGEST_DEPTH, LARGEST_DEPTH, LARGEST_DEPTH)

            collision_area = torch.sum(
                humans_bkg_depth_diff < COLLISION_DIST_THRESH, dim=[-1, -2]
            ).reshape(bkg_batch, human_batch)
            collision_judge = (
                collision_area > COLLISION_JUDGE_THRESH
            )  # * torch.sum(humans_mask_binary, dim=[-1, -2])
            if self.verbose:
                print(f"[_judge_collision] {collision_area=}")
                print(f"[_judge_collision] {collision_judge=}")
            return collision_judge

        collision_judge = _judge_collision(
            humans_image_mask_warped, humans_depth_warped, humans_height_warped, bkg_metric_depth
        )
        humans_image_mask_warped[collision_judge] = 0
        humans_depth_warped[collision_judge] = LARGEST_DEPTH
        del humans_height_map, humans_height_warped

        # blackout the background
        humans_image_mask_warped[:, :, :3] = (
            humans_image_mask_warped[:, :, :3] * humans_image_mask_warped[:, :, 3:4]
        )
        humans_depth_warped = humans_depth_warped * humans_image_mask_warped[
            :, :, 3:4
        ] + LARGEST_DEPTH * (1 - humans_image_mask_warped[:, :, 3:4])

        # sort humans from back to front (0 -> farthest)
        humans_depth_sorted_idx = torch.argsort(humans_foot_depth, dim=1, descending=True)
        humans_image_mask_sorted = humans_image_mask_warped[
            torch.arange(bkg_batch), humans_depth_sorted_idx
        ]
        humans_depth_sorted = humans_depth_warped[torch.arange(bkg_batch), humans_depth_sorted_idx]

        return humans_image_mask_sorted, humans_depth_sorted

    def harmonize(
        self, composed: Float[torch.Tensor, "b 3 h w"], mask: Float[torch.Tensor, "b 1 h w"]
    ) -> Float[torch.Tensor, "b 3 h w"]:
        batch, channel, height, width = composed.shape
        assert channel == 3, f"{composed.shape=}"
        assert mask.shape == (batch, 1, height, width), f"{mask.shape=}"

        composed_lr = F.interpolate(composed, (256, 256), mode="bilinear")
        mask_lr = F.interpolate(mask, (256, 256), mode="bilinear")
        output = self.harmonizer(composed_lr, composed, mask_lr, mask)
        output_fullres = output["images_fullres"]
        if len(output_fullres.shape) == 3:
            output_fullres = output_fullres.unsqueeze(0)  # in case b = 1
        return output_fullres

    def compose_humans_and_the_scene(
        self,
        humans_image_mask: Float[torch.Tensor, "b b_human 4 h w"],
        humans_depth: Float[torch.Tensor, "b b_human 1 h w"],
        background: Float[torch.Tensor, "b 3 h w"],
        background_depth: Float[torch.Tensor, "b 1 h w"],
        depth_temperature: float = 0.01,
        apply_harmonization: bool = True,
    ):
        batch, human_num, channel, height, width = humans_image_mask.shape
        assert channel == 4
        assert humans_depth.shape == (batch, human_num, 1, height, width)
        assert background.shape == (batch, 3, height, width)
        assert background_depth.shape == (batch, 1, height, width)

        # assume humans are sorted from back to front (0 -> farthest)
        humans_image_mask.nan_to_num_(nan=0)
        humans_depth.nan_to_num_(nan=1e10)
        # humans_valid_depth = (0 < humans_depth) * (humans_depth < LARGEST_DEPTH - 1)
        # humans_mean_depth = torch.sum(humans_depth * humans_valid_depth, dim=[2, 3, 4]) / (1 + torch.sum(humans_valid_depth, dim=[2, 3, 4]))
        # assert torch.all(humans_mean_depth[:, :-1] >= humans_mean_depth[:, 1:]), f"{humans_mean_depth=}"

        # paste humans on the background
        ret = background.clone()
        ret_mask = torch.zeros_like(background_depth)
        ret_depth = torch.clip(background_depth, 0, LARGEST_DEPTH)
        ret_label = torch.full_like(background_depth, 255, dtype=torch.uint8)
        for n in range(human_num):
            person_rgb, person_alpha = torch.split(humans_image_mask[:, n, :, :, :], [3, 1], dim=1)
            person_depth = humans_depth[:, n, :, :, :]

            if apply_harmonization:
                composed_tmp = (1 - person_alpha) * background + person_alpha * person_rgb
                harmonized_tmp = self.harmonize(composed_tmp, person_alpha)
                person_rgb = (1 - person_alpha) * person_rgb + person_alpha * harmonized_tmp
                humans_image_mask[:, n, :3, :, :] = person_rgb

            modal_mask = person_alpha * torch.sigmoid(
                (ret_depth - person_depth) / depth_temperature
            )
            ret = (1 - modal_mask) * ret + modal_mask * person_rgb
            ret_mask = torch.max(ret_mask, modal_mask)
            ret_depth = (1 - modal_mask) * ret_depth + modal_mask * person_depth
            ret_label[modal_mask > 0.5] = n

        return torch.cat([ret, ret_mask], dim=1), ret_label, humans_image_mask

    def forward(self, img: Float[torch.Tensor, "b 3 h w"]):
        batch, channel, height, width = img.shape
        assert channel == 3

        # Step1. MoGe -> focal_len & scale-invariant point cloud
        output = self.moge.infer(img)
        points = output["points"]  # (B, H, W, 3)
        depth = output["depth"]  # (B, H, W)
        # mask = output["mask"]  # (B, H, W)
        intrinsics = output["intrinsics"].clone()  # (B, 3, 3)
        intrinsics[:, 0, :] *= width
        intrinsics[:, 1, :] *= height
        cam_focal_len = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2
        del output

        # Step2. Metric3D (MoGe focal_len is better in HyperSim scenes)
        metric3dv2_depth, _ = self.get_metric_depth(
            img, cam_focal_len
        )  # NOTE: (batch, height, width)

        # Step3. Scale alignment
        scale = (
            torch.median(metric3dv2_depth.reshape(batch, -1), dim=-1).values
            / torch.median(depth.reshape(batch, -1), dim=-1).values
        )
        moge_metric_depth = depth * scale.reshape(batch, 1, 1)
        moge_metric_points = points * scale.reshape(batch, 1, 1, 1)
        del depth, points, scale

        # Step4. Ground mask
        ground_mask = self.segment_ground(img).squeeze(1) > 0  # (batch, height, width)

        # Step5. Fit a plane
        plane_coeff = []
        invalid_plane_coeff = torch.tensor(
            [1, 0, 0, torch.nan], dtype=img.dtype, device=img.device
        )
        for b in range(batch):
            g_mask = ground_mask[b]
            ground_points = moge_metric_points[b][g_mask]  # (n, 3)
            if ground_points.shape[0] == 0:
                plane_coeff.append(invalid_plane_coeff)
                if self.verbose:
                    print(f"batch{b}: ground not detected")
                continue

            # Skip RANSAC for now. Maybe necessary in the future???
            # https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
            # https://scipy.github.io/old-wiki/pages/Cookbook/RANSAC
            pa, pb, pc, pd = best_fitting_plane(ground_points.unsqueeze(0), equation=True)
            g_normal = F.normalize(
                torch.tensor([pa, pb, pc], dtype=pa.dtype, device=pa.device), dim=0
            )

            # sanity check with GeoCalib
            calib_result = self.geocalib.calibrate(img[b])
            gravity_vec = calib_result["gravity"].vec3d  # already L2 normalized
            cos_sim = torch.sum(g_normal * gravity_vec)
            plane_coeff.append(
                torch.cat([pa, pb, pc, pd], dim=-1)
                if torch.abs(cos_sim) > 0.9
                else invalid_plane_coeff
            )

            if self.verbose:
                print(f"batch{b}: {cos_sim=} => {plane_coeff[-1]}")

        plane_coeff = torch.stack(plane_coeff, dim=0)
        del ground_mask, g_mask, ground_points, moge_metric_points

        # Step6. Generate humans
        human_num = random.randint(self.max_human_num_range[0], self.max_human_num_range[1])
        if self.verbose:
            print(f"[generate_humans] {human_num=}")

        HUMAN_GEN_BATCH = 8  # batch=8 fits in GPU memory
        quotient, remainder = divmod(human_num, HUMAN_GEN_BATCH)
        humans, humans_mask = [], []
        for _ in range(quotient):
            humans.append(self.generate_human(HUMAN_GEN_BATCH, device=img.device))
            humans_mask.append(self.segment_human(humans[-1]))
        if remainder > 0:
            humans.append(self.generate_human(remainder, device=img.device))
            humans_mask.append(self.segment_human(humans[-1]))
        humans = torch.cat(humans, dim=0)
        humans_mask = torch.cat(humans_mask, dim=0)

        # Step7. Place the humans in the scene (COMPROMISE: humans_depth is constant inside each human)
        humans_image_mask, humans_depth = self.place_humans_in_the_scene(
            humans, humans_mask, moge_metric_depth, plane_coeff, intrinsics
        )
        assert humans_image_mask.shape == (batch, human_num, 4, height, width)
        assert humans_depth.shape == (batch, human_num, 1, height, width)
        del humans, humans_mask, plane_coeff, cam_focal_len, intrinsics

        # Step8. Compose the humans and the scene
        composed_rgba, composed_label, humans_image_mask = self.compose_humans_and_the_scene(
            humans_image_mask,
            humans_depth,
            img,
            moge_metric_depth.reshape(batch, 1, height, width),
            depth_temperature=0.01,
            apply_harmonization=True,
        )
        assert composed_rgba.shape == (batch, 4, height, width), f"{composed_rgba.shape=}"
        assert composed_label.shape == (batch, 1, height, width), f"{composed_label.shape=}"
        assert composed_label.dtype == torch.uint8, f"{composed_label.dtype=}"
        del humans_depth, moge_metric_depth

        return composed_rgba, composed_label, humans_image_mask
