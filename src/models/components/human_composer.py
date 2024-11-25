import sys

import cv2
import depth_pro
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from geocalib import GeoCalib
from groundingdino.util.inference import Model as GroundingDINOModule
from groundingdino.util.utils import get_phrases_from_posmap
from jaxtyping import Float
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

from third_party.MaGGIe.maggie.network.arch import MaGGIe
from third_party.PCTNet.iharm.inference.utils import load_model
from third_party.StyleGAN_Human import dnnlib, legacy

STYLEGAN_HUMAN_PKL = (
    "/home/ryotaro/github/StyleGAN-Human/pretrained_models/stylegan_human_v2_1024.pkl"
)
GROUNDING_DINO_CFG = "/home/ryotaro/github/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CKPT = "/home/ryotaro/github/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
DEPTHPRO_CKPT = "/home/ryotaro/github/ml-depth-pro/checkpoints/depth_pro.pt"
HARMONIZER_WEIGHT = (
    "/home/ryotaro/my_works/HoGe/third_party/PCTNet/pretrained_models/PCTNet_ViT.pth"
)


class HumanCompositionError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class HumanComposer(nn.Module):
    def __init__(
        self,
        use_depthpro: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.use_depthpro = use_depthpro
        self.verbose = verbose

        # StyleGAN-Human
        with dnnlib.util.open_url(STYLEGAN_HUMAN_PKL) as f:
            self.stylegan_human = legacy.load_network_pkl(f)["G_ema"]

        # YOLO11 Human Segmentation
        self.human_segmetor = YOLO("yolo11n-seg.pt").cpu()
        self.human_segmetor.train = lambda yolo_self: None

        # MaGGIe Human Matting
        self.matting_model = MaGGIe.from_pretrained("chuonghm/maggie-image-him50k-cvpr24").cpu()

        # GroundedSAM
        self.grounding_dino_model = GroundingDINOModule(
            model_config_path=GROUNDING_DINO_CFG,
            model_checkpoint_path=GROUNDING_DINO_CKPT,
            device="cpu",
        ).model
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

        if use_depthpro:
            # DepthPro
            cfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
            cfg.checkpoint_uri = DEPTHPRO_CKPT
            self.depth_pro_model, _ = depth_pro.create_model_and_transforms()
        else:
            # GeoCalib & Metric3Dv2
            self.geocalib = GeoCalib()
            self.metric3dv2 = torch.hub.load(
                "yvanyin/metric3d", "metric3d_vit_small", pretrain=True
            )

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
        # ground detection
        CLASSES = [
            "Ground",
            "Floor",
            "Carpet",
        ]
        BOX_THRESHOLD = 0.25
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8

        # size so the shorter edge size is 800 but the longer edge size doesn't exceed 1333
        batch, channel, height, width = img.shape
        assert channel == 3

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device).reshape(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device).reshape(
            1, 3, 1, 1
        )
        img_norm = (img - mean) / std

        if width < height:
            ow = min(800, 1333 * width // height)
            oh = height * ow // width
        else:
            oh = min(800, 1333 * height // width)
            ow = width * oh // height
        img_preprocessed = F.interpolate(img_norm, (oh, ow), mode="bilinear")

        caption = ". ".join(CLASSES) + "."
        with torch.no_grad():
            outputs = self.grounding_dino_model(
                img_preprocessed, captions=[caption for _ in range(batch)]
            )

        # nq => num of object queries, 256 => object class description (in case the caption contain different class names)
        prediction_confs = (
            outputs["pred_logits"].sigmoid().max(dim=-1)[0]
        )  # prediction_logits.shape = (b, nq, 256) -> (b, nq)
        prediction_boxes = outputs["pred_boxes"]  # prediction_boxes.shape = (b, nq, 4)
        prediction_boxes *= torch.tensor(
            [[[width, height, width, height]]], dtype=img.dtype, device=img.device
        )  # normalized -> absolute
        prediction_boxes = torchvision.ops.box_convert(
            prediction_boxes.reshape(-1, 4), in_fmt="cxcywh", out_fmt="xyxy"
        ).reshape(batch, -1, 4)

        # NMS for each image
        sam_bbox_input = []
        for confs_init, boxes_init in zip(prediction_confs, prediction_boxes):
            # extract valid bboxes
            masks = confs_init > BOX_THRESHOLD
            confs = confs_init[masks]  # (n)
            boxes = boxes_init[masks]  # (n, 4)

            if not torch.any(masks):
                raise HumanCompositionError(
                    f"Ground not detected (DINO_confs.max()={confs_init.max().cpu()})"
                )

            if self.verbose:
                print(f"[HumanComposer] segment_ground: DINO_{confs=}")

            # NMS post process
            # print(f"Before NMS: {len(boxes)} boxes")
            nms_idx = torchvision.ops.nms(boxes, confs, NMS_THRESHOLD)
            boxes = boxes[nms_idx]
            # print(f"After NMS: {len(boxes)} boxes")

            # SAM input format
            sam_bbox_input.append(boxes)

        # SAM segmentation
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 1. set_image_batch
            self.sam_predictor.reset_predictor()
            self.sam_predictor._orig_hw = [(height, width) for _ in range(batch)]
            img_batch = F.interpolate(
                img_preprocessed,
                (self.sam_predictor.model.image_size, self.sam_predictor.model.image_size),
                mode="bilinear",
            )

            backbone_out = self.sam_predictor.model.forward_image(img_batch)
            _, vision_feats, _, _ = self.sam_predictor.model._prepare_backbone_features(
                backbone_out
            )
            # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
            if self.sam_predictor.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.sam_predictor.model.no_mem_embed

            feats = [
                feat.permute(1, 2, 0).view(batch, -1, *feat_size)
                for feat, feat_size in zip(
                    vision_feats[::-1], self.sam_predictor._bb_feat_sizes[::-1]
                )
            ][::-1]
            self.sam_predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
            self.sam_predictor._is_image_set = True
            self.sam_predictor._is_batch = True

            # 2. predict
            result_masks = []
            for img_idx in range(batch):
                mask_input, unnorm_coords, labels, unnorm_box = self.sam_predictor._prep_prompts(
                    None,  # point_coords,
                    None,  # point_labels,
                    sam_bbox_input[img_idx],
                    None,  # mask_input,
                    normalize_coords=True,
                    img_idx=img_idx,
                )
                masks, iou_predictions, low_res_masks = self.sam_predictor._predict(
                    unnorm_coords,
                    labels,
                    unnorm_box,
                    mask_input,
                    multimask_output=True,
                    return_logits=False,
                    img_idx=img_idx,
                )
                if self.verbose:
                    print(f"[HumanComposer] segment_ground: SAM2_{iou_predictions=}")

                index = torch.argmax(iou_predictions, dim=1)
                masks_highest_score = masks[torch.arange(len(index)), index]
                result_masks.append(
                    torch.max(masks_highest_score, dim=0)[0]
                )  # union of different object masks

                if torch.mean(result_masks[-1].float()) > 0.95:
                    raise HumanCompositionError(
                        f"Too large ground mask (masks.mean()={torch.mean(result_masks[-1].float()).cpu()})"
                    )

        ground_mask = torch.stack(result_masks, dim=0).float().unsqueeze(1)
        ground_mask = F.interpolate(ground_mask, (height, width), mode="bilinear")
        return ground_mask  # , sam_bbox_input

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
        if self.use_depthpro:
            prediction = self.depth_pro_model.infer(2 * bkg - 1, f_px=None)
            metric_depth = prediction["depth"].reshape(batch, h, w)  # Depth in [m].
            cam_focal_len = prediction["focallength_px"]
        else:
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
        ret, ret_mask, metric_depth = self.configure(human, segmask, img, ground_mask)
        ret = self.harmonize(ret, ret_mask)
        return ret, ret_mask, ground_mask, metric_depth
