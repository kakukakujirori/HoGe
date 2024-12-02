import torch
import torch.nn as nn
import torch.nn.functional as F
import utils3d
from geocalib import GeoCalib
from jaxtyping import Float
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
    TexturesUV,
)
from pytorch3d.renderer.blending import BlendParams, hard_rgb_blend
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from ultralytics import YOLO

from src.models.components.moge_mesh import image_mesh
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
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

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


# >>> Human Renderer Classes >>>


class SimpleShaderWithMask(torch.nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)  # [N, H, W, K, 4]
        return texels[..., 0, :]  # (N, H, W, 4) RGBA image


class AlphaCompositionShader(torch.nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        # Get pixel colors and alphas from the texture
        pixel_colors = meshes.sample_textures(fragments)  # [N, H, W, K, 4]

        # Extract alpha values
        alphas = pixel_colors[..., 3]

        # Perform alpha compositing
        # This is a basic over-compositing approach
        # Assumes front-to-back rendering order
        composite_image = torch.ones_like(pixel_colors[..., 0, :3])
        composite_image[..., 0] = self.blend_params.background_color[0]
        composite_image[..., 1] = self.blend_params.background_color[1]
        composite_image[..., 2] = self.blend_params.background_color[2]

        # Iterate through the K samples (typically from rasterization)
        for k in range(pixel_colors.shape[3]):
            # Current layer's color and alpha
            layer_color = pixel_colors[..., k, :3]
            layer_alpha = alphas[..., k]

            # Blend with previous composite
            composite_image = layer_color * layer_alpha[..., None] + composite_image * (
                1 - layer_alpha[..., None]
            )

        return composite_image


class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf[..., 0]


# <<< Human Renderer Classes <<<


class HumanComposer3D(nn.Module):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__()
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
        self, batch: int, truncation_psi: float = 1.0, noise_mode: str = "const", device="cuda"
    ):
        assert noise_mode in ["const", "random", "none"]
        label = torch.zeros([batch, self.stylegan_human.c_dim], device=device)
        z = torch.randn(batch, self.stylegan_human.z_dim, dtype=torch.float32, device=device)
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

        # matting
        batch = {"image": human_resized.unsqueeze(1), "mask": segmask_resized.unsqueeze(1)}
        output = self.matting_model(batch)

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

    def place_a_human_in_the_scene(
        self,
        human: Float[torch.Tensor, "b 3 h w"],
        human_mask: Float[torch.Tensor, "b 1 h w"],
        bkg_metric_depth: Float[torch.Tensor, "b h_bkg w_bkg"],
        plane_coeff: Float[torch.Tensor, "b 4"],
        cam_focal_len: float,
        return_mesh: bool = True,  # if False, human point cloud is returned
    ) -> Float[torch.Tensor, "b h w 3"]:
        human_alpha_ch_last = torch.cat([human, human_mask], dim=1).permute(0, 2, 3, 1)
        batch, human_height, human_width, _ = human_alpha_ch_last.shape
        PA = plane_coeff[:, 0]
        PB = plane_coeff[:, 1]
        PC = plane_coeff[:, 2]
        PD = plane_coeff[:, 3]
        ground_normal = plane_coeff[:, :3]

        foot_pos_in_fg, current_human_pixel_height = (
            __class__.get_human_foot_pixel_coord_and_height(human_mask, index="xy")
        )

        human_actual_height = torch.normal(
            1.7, 0.07, (batch,), dtype=human.dtype, device=human.device
        )
        human_min_depth = 1.0
        human_max_depth = torch.max(bkg_metric_depth.reshape(batch, -1), dim=1).values.clamp(
            human_min_depth, 5.0
        )
        human_depth = human_min_depth + torch.rand_like(human_actual_height) * (
            human_max_depth - human_min_depth
        )
        if self.verbose:
            print(f"[place_a_human_in_the_scene] {human_actual_height=}\n{human_depth=}")
        """Put a human on the intersection line of ax+by+cz+d=0 (plane_eq) and z=human_depth.
        Note that nearly a=c=0 in most cases.

        The intersection line is parametrized by (x, (d - c*human_depth - ax) / b, human_depth)
        The free variable x is constrained by the camera FoV.
        """
        # foot position in 3D space
        bkg_batch, bkg_height, bkg_width = bkg_metric_depth.shape
        assert batch == bkg_batch
        foot_pos_in_bg_x_max = (bkg_width / 2) * human_depth / cam_focal_len
        foot_pos_in_bg_x = (torch.rand_like(human_depth) * 2 - 1) * foot_pos_in_bg_x_max
        foot_pos_in_bg_y = (-PD - PC * human_depth - PA * foot_pos_in_bg_x) / PB
        foot_pos_in_bg = torch.stack([foot_pos_in_bg_x, foot_pos_in_bg_y, human_depth], dim=-1)

        # define the whole body points in 3D (metric depth is correct in all directions x, y, andz)
        scale_factor = human_actual_height / current_human_pixel_height

        # define the whole body points in 3D plane, PARPENDICULAR TO THE PRINCIPAL AXIS for ease
        # (NOTE: metric depth is correct in all directions x, y, and z)
        scale_factor = human_actual_height / current_human_pixel_height
        human_pixels = (
            torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(0, human_width - 1, human_width, device=human.device),
                        torch.linspace(0, human_height - 1, human_height, device=human.device),
                    ],
                    indexing="xy",
                ),
                dim=-1,
            )
            .reshape(1, -1, 2)
            .repeat(batch, 1, 1)
        )

        human_points_plane_xy = (
            human_pixels - foot_pos_in_fg.reshape(batch, 1, 2)
        ) * scale_factor.reshape(batch, 1, 1)
        human_points_plane_z = torch.zeros_like(human_points_plane_xy[..., 0:1])
        human_points_plane = torch.cat([human_points_plane_xy, human_points_plane_z], dim=-1)
        human_points = foot_pos_in_bg.reshape(batch, 1, 3) + human_points_plane

        # rotate the human plane so the body axis aligns with the ground normal with the foot point fixed
        initial_normal = torch.zeros(
            batch, 3, dtype=ground_normal.dtype, device=ground_normal.device
        )
        initial_normal[:, 1] = torch.sign(ground_normal[:, 1])
        target_normal = F.normalize(ground_normal, dim=1)
        rotation_axis = F.normalize(torch.linalg.cross(initial_normal, target_normal), dim=1)
        angle = torch.acos(torch.sum(initial_normal * target_normal, dim=1))
        rotmat = axis_angle_to_matrix(angle.reshape(batch, 1) * rotation_axis)
        human_points = (
            rotmat.reshape(batch, 1, 3, 3)
            @ (human_points - foot_pos_in_bg.reshape(batch, 1, 3)).reshape(batch, -1, 3, 1)
        ).squeeze(-1) + foot_pos_in_bg.reshape(batch, 1, 3)

        if not return_mesh:
            return human_points.reshape(batch, human_height, human_width, 3)

        # pack into a mesh
        faces_list = []
        verts_list = []
        vert_uv_list = []
        for b in range(batch):  # TODO: OPTIMIZE image_mesh()
            faces, vertices, vertex_colors, vertex_uvs = image_mesh(
                human_points[b].reshape(human_height, human_width, 3),
                human_alpha_ch_last[b],
                utils3d.torch.image_uv(
                    height=human_height, width=human_width, device=human.device
                ),
                mask=None,  # (mask & ~depth_edge) if invalid_mesh_color is None else mask,
                tri=True,
            )
            vertex_uvs[:, 1] = 1 - vertex_uvs[:, 1]
            faces_list.append(faces)
            verts_list.append(vertices)
            vert_uv_list.append(vertex_uvs)

        faces_list = torch.stack(faces_list)
        verts_list = torch.stack(verts_list)
        vert_uv_list = torch.stack(vert_uv_list)

        # Create a textures object
        tex_list = TexturesUV(
            maps=human_alpha_ch_last,
            faces_uvs=faces_list,
            verts_uvs=vert_uv_list,
            align_corners=False,
        )
        return Meshes(verts=verts_list, faces=faces_list, textures=tex_list)

    @staticmethod
    def render_human_mesh(
        human_mesh: Meshes,
        intrinsics: Float[torch.Tensor, "b 3 3"],
        render_height: int,
        render_width: int,
    ) -> tuple[Float[torch.Tensor, "b h w 4"], Float[torch.Tensor, "b h w"]]:
        batch = intrinsics.shape[0]
        device = intrinsics.device

        # OpenCV -> PyTorch3D
        xy_flipvec = torch.tensor([-1, -1, 1], dtype=intrinsics.dtype, device=device)
        xy_flipmat_3x3 = torch.diag(xy_flipvec).unsqueeze(0)
        pose = torch.eye(3, device=device).reshape(1, 3, 3).repeat(batch, 1, 1)
        R = (xy_flipmat_3x3 @ pose[:, :3, :3]).permute(0, 2, 1)

        cameras = PerspectiveCameras(
            focal_length=intrinsics.diagonal(dim1=1, dim2=2)[:, :2],
            principal_point=intrinsics[:, :2, 2],
            R=R,
            T=torch.zeros(1, 3, device=device).repeat(batch, 1),
            device=device,
            in_ndc=False,
            image_size=[(render_height, render_width) for _ in range(batch)],
        )

        raster_settings = RasterizationSettings(
            image_size=(render_height, render_width),
            blur_radius=1e-12,
            faces_per_pixel=1,
            bin_size=None,
            cull_backfaces=False,  # This will ignore back-facing polygons
            perspective_correct=True,
        )

        renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SimpleShaderWithMask(
                device=device, blend_params=BlendParams(background_color=(0, 0, 0))
            ),
        )

        human_rendered, human_rendered_depth = renderer(human_mesh)

        return human_rendered, human_rendered_depth

    def harmonize(
        self, composed: Float[torch.Tensor, "b 3 h w"], mask: Float[torch.Tensor, "b 1 h w"]
    ) -> Float[torch.Tensor, "b 3 h w"]:
        composed_lr = F.interpolate(composed, (256, 256), mode="bilinear")
        mask_lr = F.interpolate(mask, (256, 256), mode="bilinear")
        output = self.harmonizer(composed_lr, composed, mask_lr, mask)
        output_fullres = output["images_fullres"]
        if len(output_fullres.shape) == 3:
            output_fullres = output_fullres.unsqueeze(0)  # in case b = 1
        return output_fullres

    def forward(self, img: Float[torch.Tensor, "b 3 h w"]):
        batch, channel, height, width = img.shape
        assert channel == 3

        # Step1. MoGe -> focal_len & scale-invariant point cloud
        output = self.moge.infer(img)
        points = output["points"]  # (B, H, W, 3)
        depth = output["depth"]  # (B, H, W)
        mask = output["mask"]  # (B, H, W)
        intrinsics = output["intrinsics"].clone()  # (B, 3, 3)
        intrinsics[:, 0, :] *= width
        intrinsics[:, 1, :] *= height
        cam_focal_len = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2

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

        # Step4. Ground mask
        ground_mask = self.segment_ground(img).squeeze(1) > 0  # (batch, height, width)

        # Step5. Fit a plane
        plane_coeff = []
        for b in range(batch):
            g_mask = ground_mask[b]
            if torch.any(g_mask):
                ground_points = moge_metric_points[b][g_mask]

                # Skip RANSAC for now. Maybe necessary in the future???
                # https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
                # https://scipy.github.io/old-wiki/pages/Cookbook/RANSAC
                pa, pb, pc, pd = best_fitting_plane(ground_points.unsqueeze(0), equation=True)
                plane_coeff.append(torch.cat([pa, pb, pc, pd], dim=-1))
            else:
                plane_coeff.append(
                    torch.tensor([1, 0, 0, torch.nan], dtype=img.dtype, device=img.device)
                )
        plane_coeff = torch.stack(plane_coeff, dim=0)

        # Step6. Generate a human
        human = self.generate_human(batch=img.shape[0], device=img.device)
        human_mask = self.segment_human(human)

        # Step7. Place the human in the scene
        human_mesh = self.place_a_human_in_the_scene(
            human, human_mask, moge_metric_depth, plane_coeff, cam_focal_len, return_mesh=True
        )

        # Step8. Render the human
        human_rendered, human_rendered_depth = self.render_human_mesh(
            human_mesh,
            intrinsics,
            height,
            width,
        )

        # Step9. Compose the human and the scene
        img_ch_last = img.permute(0, 2, 3, 1)
        ret_mask = human_rendered[..., 3:4]
        ret = (1 - ret_mask) * img_ch_last + ret_mask * human_rendered[..., :3]

        bg_forefront_mask = (moge_metric_depth < human_rendered_depth).unsqueeze(-1)
        ret = torch.where(bg_forefront_mask, img_ch_last, ret)
        ret_mask[bg_forefront_mask] = 0

        # Step10. Harmonize (NOTE: back to channel first)
        ret = ret.permute(0, 3, 1, 2)
        ret_mask = ret_mask.reshape(batch, 1, height, width)
        ret = self.harmonize(ret, ret_mask)

        return ret, ret_mask
