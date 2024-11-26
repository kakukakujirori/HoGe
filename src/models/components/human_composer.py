import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from geocalib import GeoCalib
from jaxtyping import Float
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from ultralytics import YOLO

from third_party.MaGGIe.maggie.network.arch import MaGGIe
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


class HumanComposer(nn.Module):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose

        # StyleGAN-Human
        with dnnlib.util.open_url(STYLEGAN_HUMAN_PKL) as f:
            self.stylegan_human = legacy.load_network_pkl(f)["G_ema"]

        # YOLO11 Human Segmentation
        self.human_segmetor = YOLO("yolo11n-seg.pt").cpu()
        self.human_segmetor.train = lambda yolo_self: None

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

        # GeoCalib & Metric3Dv2
        self.geocalib = GeoCalib()
        self.metric3dv2 = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)

        # Harmonizer
        model_type = "ViT_pct"
        self.harmonizer = load_model(model_type, HARMONIZER_WEIGHT, verbose=False)

        # compile
        # self.compile()

    def compile(self):
        self.stylegan_human = torch.compile(self.stylegan_human)
        self.human_segmetor.model = torch.compile(self.human_segmetor.model)
        self.matting_model = torch.compile(self.matting_model)
        self.grounding_dino_model = torch.compile(self.grounding_dino_model)
        self.sam_predictor.model = torch.compile(self.sam_predictor.model)
        # self.depth_pro_model = torch.compile(self.depth_pro_model)
        self.metric3dv2 = torch.compile(self.metric3dv2)

    def generate_human(
        self, batch: int, truncation_psi: float = 1.0, noise_mode: str = "const", device="cuda"
    ):
        assert noise_mode in ["const", "random", "none"]
        label = torch.zeros([batch, self.stylegan_human.c_dim], device=device)
        z = torch.from_numpy(np.random.randn(batch, self.stylegan_human.z_dim)).to(device)
        w = self.stylegan_human.mapping(z, label, truncation_psi=truncation_psi)
        human = self.stylegan_human.synthesis(w, noise_mode=noise_mode, force_fp32=True)
        human = torch.clip((human + 1) / 2, 0, 1)  # (-1, 1) -> (0, 1)
        return human

    def segment_human(
        self, human: Float[torch.Tensor, "b 3 h w"]
    ) -> Float[torch.Tensor, "b 1 h w"]:
        b, c, ori_h, ori_w = human.shape
        assert c == 3

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

        # segment (NOTE: Ultralytics YOLO implicitly assumes that the input 'tensor' longer edge length is 640)
        human_segsize, _, _ = _resize(human, 640, False)
        results = self.human_segmetor(human_segsize, classes=[0], retina_masks=True)
        segmask = torch.stack(
            [torch.max(ret.masks.data, dim=0, keepdim=True)[0] for ret in results], dim=0
        )
        if segmask.shape[0] != b:
            raise RuntimeError

        # preprocess matting image
        human_resized, h, w = _resize(
            human, 576, to_short_size=True, multiple_of=64, padding_val=1
        )
        segmask_resized, _, _ = _resize(
            segmask, 576, to_short_size=True, multiple_of=64, padding_val=0
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

    def segment_ground(
        self, img: Float[torch.Tensor, "b 3 h w"]
    ) -> Float[torch.Tensor, "b 1 h w"]:
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

        # In ADE20K
        ground_mask = (
            (predicted_semantic_map == 3)  # 3 => floor
            + (predicted_semantic_map == 6)  # 6 => road, route
            + (predicted_semantic_map == 9)  # 9 => grass
            + (predicted_semantic_map == 11)  # 11 => sidewalk, pavement
            + (predicted_semantic_map == 13)  # 13 => earth, ground
            + (predicted_semantic_map == 28)  # 28 => rug
            + (predicted_semantic_map == 53)  # 53 => stairs
            + (predicted_semantic_map == 59)  # 59 => stairway, staircase
            + (predicted_semantic_map == 96)  # 96 => escalator, moving staircase, moving stairway
            + (predicted_semantic_map == 121)  # 121 => step, stair
        )
        return ground_mask.float()

    def get_metric_depth(self, img: Float[torch.Tensor, "b 3 h w"]):
        batch, channel, height, width = img.shape
        assert channel == 3

        # calibration
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

        return pred_depth.squeeze(1), cam_focal_len  # align with the DepthPro output

    def configure(
        self,
        human: Float[torch.Tensor, "b 3 h_human w_human"],
        human_mask: Float[torch.Tensor, "b 1 h_human w_human"],
        bkg: Float[torch.Tensor, "b 3 h w"],
        ground_mask: Float[torch.Tensor, "b 1 h w"],
    ):
        device = human.device
        b_human, c_human, h_human, w_human = human.shape
        batch, c, h, w = bkg.shape
        assert b_human == batch
        assert c_human == c == 3
        assert human_mask.shape == (batch, 1, h_human, w_human)
        assert ground_mask.shape == (batch, 1, h, w)

        # metric depth & focal_len
        metric_depth, cam_focal_len = self.get_metric_depth(bkg)
        if self.verbose:
            print(f"[HumanComposer] configure: {cam_focal_len=}")

        # select human position and size
        true_indices = ground_mask.nonzero(as_tuple=False)  # (N, 4)
        batch_counts = true_indices[:, 0].bincount(minlength=ground_mask.shape[0])  # (B,)
        ground_exists = batch_counts > 0
        assert torch.all(ground_exists)
        rand_indices = (
            torch.cat([torch.tensor([0], device=device), batch_counts.cumsum(0)[:-1]])
            + (torch.rand(ground_mask.shape[0], device=device) * batch_counts.float()).long()
        ) * ground_exists
        foot_pos_in_bg = torch.index_select(true_indices[:, 2:], 0, rand_indices)  # (B, 2)
        human_depth = metric_depth[
            torch.arange(batch, device=device), foot_pos_in_bg[:, 0], foot_pos_in_bg[:, 1]
        ]
        human_actual_height = torch.normal(1.7, 0.07, (batch,), dtype=bkg.dtype, device=bkg.device)
        human_pixel_height = human_actual_height * cam_focal_len / human_depth

        # human foot position
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
            human_mask_vertical.reshape(batch, h_human), dim=1
        )  # NOTE: Should be more sophisticated
        assert foot_pos_in_fg.shape == (batch, 2)

        # base result
        ret = bkg.clone()
        ret_mask = torch.zeros_like(ground_mask)

        # resize the human
        scale_factor = human_pixel_height / current_human_pixel_height
        foot_pose_in_fg_resize = torch.round(foot_pos_in_fg * scale_factor.reshape(batch, 1)).to(
            torch.int32
        )
        for b in range(batch):
            human_resized = F.interpolate(
                human[b : b + 1], scale_factor=scale_factor[b].item(), mode="bilinear"
            ).squeeze(0)
            human_mask_resized = F.interpolate(
                human_mask[b : b + 1], scale_factor=scale_factor[b].item(), mode="bilinear"
            ).squeeze(0)

            # these can be outside the image region
            bbox_up = foot_pos_in_bg[b][0] - foot_pose_in_fg_resize[b][0]
            bbox_down = foot_pos_in_bg[b][0] + (
                human_resized.shape[-2] - foot_pose_in_fg_resize[b][0]
            )
            bbox_left = foot_pos_in_bg[b][1] - foot_pose_in_fg_resize[b][1]
            bbox_right = foot_pos_in_bg[b][1] + (
                human_resized.shape[-1] - foot_pose_in_fg_resize[b][1]
            )

            # corresponding foreground
            bbox_fg_up = max(0, -bbox_up)
            bbox_fg_down = human_resized.shape[-2] - max(0, bbox_down - h)
            bbox_fg_left = max(0, -bbox_left)
            bbox_fg_right = human_resized.shape[-1] - max(0, bbox_right - w)

            # limit bbox inside the image region
            bbox_up = max(0, bbox_up)
            bbox_down = min(h, bbox_down)
            bbox_left = max(0, bbox_left)
            bbox_right = min(w, bbox_right)

            human_cropped = human_resized[:, bbox_fg_up:bbox_fg_down, bbox_fg_left:bbox_fg_right]
            human_mask_cropped = human_mask_resized[
                :, bbox_fg_up:bbox_fg_down, bbox_fg_left:bbox_fg_right
            ]
            bg_cropped = bkg[b, :, bbox_up:bbox_down, bbox_left:bbox_right]

            # overlay
            ret[b, :, bbox_up:bbox_down, bbox_left:bbox_right] = (
                1 - human_mask_cropped
            ) * bg_cropped + human_mask_cropped * human_cropped
            ret_mask[b, :, bbox_up:bbox_down, bbox_left:bbox_right] = human_mask_cropped

            bg_forefront_mask = (metric_depth[b] < human_depth[b]).unsqueeze(0)
            ret[b] = torch.where(bg_forefront_mask, bkg[b], ret[b])
            ret_mask[b][bg_forefront_mask] = 0

        return ret, ret_mask, metric_depth

    def harmonize(
        self, composed: Float[torch.Tensor, "b 3 h w"], mask: Float[torch.Tensor, "b 1 h w"]
    ):
        composed_lr = F.interpolate(composed, (256, 256), mode="bilinear")
        mask_lr = F.interpolate(mask, (256, 256), mode="bilinear")
        output = self.harmonizer(composed_lr, composed, mask_lr, mask)
        output_fullres = output["images_fullres"]
        if len(output_fullres.shape) == 3:
            output_fullres = output_fullres.unsqueeze(0)  # in case b = 1
        return output_fullres

    def forward(
        self, img: Float[torch.Tensor, "b c h w"]
    ) -> tuple[Float[torch.Tensor, "b c h w"], Float[torch.Tensor, "b c h w"]]:
        human = self.generate_human(batch=img.shape[0], device=img.device)
        segmask = self.segment_human(human)
        ground_mask = self.segment_ground(img)
        if torch.all(ground_mask < 0.01):
            raise HumanCompositionError("No ground found")
        ret, ret_mask, metric_depth = self.configure(human, segmask, img, ground_mask)
        ret = self.harmonize(ret, ret_mask)
        return ret, ret_mask, ground_mask, metric_depth
