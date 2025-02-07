import os
from typing import Any, Dict, Tuple

import torch
import utils3d
from einops import rearrange
from jaxtyping import Float
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes

from src.models.components.hoge_model import HoGeModel
from src.models.components.render_mesh import RenderMesh
from src.models.losses.optimal_alignment_loss import OptimalAlignmentLoss


class MeshHoGeModule(LightningModule):
    """A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: HoGeModel,
        criterion_list: list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # gather_layers_before_inference: bool,
        log_images_every_n_steps: int = -1,
        compile: bool = True,
    ) -> None:
        """Initialize a `MeshHoGeModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.mesh_renderer = RenderMesh(layer_num=net.max_points_per_ray, void_alpha=None)

        # load MoGe weight
        self.net = net
        self.net.load_pretrained_moge()

        # loss functions
        self.criterion_list = torch.nn.ModuleList(criterion_list)

        # logging
        if log_images_every_n_steps > 0:
            self.log_images_every_n_steps = log_images_every_n_steps
        else:
            self.log_images_every_n_steps = (
                self.trainer.estimated_stepping_batches
            )  # huge number so only batch_idx=0 is recorded

    def forward(
        self, image: Float[torch.Tensor, "b c h w"], invalid_mask: Float[torch.Tensor, "b c h w"]
    ) -> dict[str, torch.Tensor]:
        return self.net(image, invalid_mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def render_mesh(
        self,
        meshes: Meshes,
        image: Float[torch.Tensor, "b h w 3"],
        focal_len: float,
    ) -> tuple[
        Float[torch.Tensor, "b h w layers 4"],
        Float[torch.Tensor, "b h w layers 3"],
    ]:
        batch, height, width, channel = image.shape
        assert channel == 3

        intrinsics = torch.eye(3, device=image.device).reshape(1, 3, 3).repeat(
            batch, 1, 1
        ) * focal_len.reshape(batch, 1, 1)
        intrinsics[:, 0, 2] = width / 2
        intrinsics[:, 1, 2] = height / 2
        camera = PerspectiveCameras(
            focal_length=focal_len.reshape(batch, 1).float(),
            principal_point=torch.tensor([width / 2, height / 2]).reshape(1, 2).expand(batch, -1),
            R=torch.diag(torch.tensor([-1, -1, 1], device=image.device))
            .reshape(1, 3, 3)
            .expand(batch, -1, -1),
            T=torch.zeros(batch, 3, device=image.device),
            device=image.device,
            in_ndc=False,
            image_size=[(height, width) for _ in range(batch)],
        )

        # PyTorch3D doesn't support float16
        with torch.autocast(device_type="cuda", dtype=torch.float32):

            texels_rendered, depths_rendered = self.mesh_renderer(
                meshes, camera
            )  # (b, h, w, layers, 3), (b, h, w, layers)

            # add alpha channel
            texels_rendered = torch.cat(
                [
                    texels_rendered,
                    (depths_rendered.isfinite() * (depths_rendered > 0)).float().unsqueeze(-1),
                ],
                dim=-1,
            )

            # unproject depths_rendered to points_rendered
            uv_coord = utils3d.torch.image_uv(
                width=width, height=height, dtype=image.dtype, device=image.device
            )
            points_rendered = torch.cat([uv_coord, torch.ones_like(uv_coord[..., :1])], dim=-1)
            points_rendered = points_rendered @ torch.inverse(
                intrinsics.reshape(batch, 1, 3, 3)
            ).transpose(
                -2, -1
            )  # (h, w, 3) @ (b, 1, 3, 3) => (b, h, w, 3)
            points_rendered = points_rendered.reshape(
                batch, height, width, 1, 3
            ) * depths_rendered.reshape(
                batch, height, width, -1, 1
            )  # (b, h, w, 1, 3) * (b, h, w, layers, 1) => (b, h, w, layers, 3)

        return texels_rendered, points_rendered

    def model_step(self, batch: dict):
        composite_image = batch["composite_image"]
        b, h, w, c = composite_image.shape
        layer_num = self.net.max_points_per_ray
        assert c == 3

        # generate pseudo GT and inputs
        texels_rendered, points_rendered = self.render_mesh(
            batch["meshes"], batch["composite_image"], batch["focal_len"]
        )

        # if self.hparams.gather_layers_before_inference:
        #     points_rendered, texels_rendered = OptimalAlignmentLoss.gather_valid_points(
        #         points_rendered,
        #         texels_rendered,
        #         mask_thresh=0.5,  # NOTE: sky region is 'valid' here
        #     )

        input_dict = {
            "composite_image": composite_image,
            "meshes": batch["meshes"],
            "texels_rendered": texels_rendered,
            "points_rendered": points_rendered,
        }

        # HoGe inference
        output_dict = self.forward(
            image=rearrange(composite_image, "b h w c -> b c h w"),
            invalid_mask=None,  # NOTE: sky region is 'valid' here
        )

        # calculate losses
        loss_dict = {}
        for loss_func in self.criterion_list:
            loss_key = type(loss_func).__name__
            loss_val = loss_func(input_dict, output_dict)
            if isinstance(loss_val, dict):
                for key, val in loss_val.items():
                    # assert torch.isfinite(val), f"{loss_key}[{key}]: {val}"
                    loss_dict[f"{loss_key}[{key}]"] = val
            else:
                # assert torch.isfinite(loss_val), f"{loss_key}: {loss_val}"
                loss_dict[loss_key] = loss_val
        loss_dict["total_loss"] = sum(loss_dict.values())

        return loss_dict, input_dict, output_dict

    def training_step(self, batch: Float[torch.Tensor, "b h w 3"], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_dict, input_dict, output_dict = self.model_step(batch)

        for loss_key, loss_val in loss_dict.items():
            self.log(
                f"train/{loss_key}",
                loss_val,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )

        if batch_idx % self.log_images_every_n_steps == 0 and self.logger.experiment:
            self.log_images(input_dict, output_dict, name="train_images", step=self.global_step)

        # return loss or backpropagation will fail
        return loss_dict["total_loss"]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss_dict, input_dict, output_dict = self.model_step(batch)

        for loss_key, loss_val in loss_dict.items():
            self.log(
                f"val/{loss_key}",
                loss_val,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        if batch_idx == 0 and self.logger.experiment:
            self.log_images(input_dict, output_dict, name="val_images", step=self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss_dict, input_dict, output_dict = self.model_step(batch)

        for loss_key, loss_val in loss_dict.items():
            self.log(
                f"test/{loss_key}",
                loss_val,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        if batch_idx == 0 and self.logger.experiment:
            self.log_images(input_dict, output_dict, name="test_images", step=self.current_epoch)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def log_images(
        self,
        input_dict: dict[torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        name: str,
        step: int,
    ) -> None:

        # only record the first data in a batch
        batch_idx = 0

        # inputs
        image_original = input_dict["composite_image"][batch_idx]  # (h, w, 3)
        # points_original = input_dict["points_original"][batch_idx]  # (h, w, 3)
        texels_rendered = input_dict["texels_rendered"][batch_idx]  # (h, w, layers, 4)
        points_rendered = input_dict["points_rendered"][batch_idx]  # (h, w, layers, 3)

        # normalize images (NOTE: invalid_mask NOT taken into account)
        image_original = rearrange(image_original, "h w c -> () h w c", c=3)
        texels_rendered = rearrange(texels_rendered, "h w layers c -> layers h w c", c=4)
        image_all = torch.cat([image_original, texels_rendered[..., :3]], dim=0)
        image_all = torch.clip(255 * image_all, 0, 255).to(torch.uint8).cpu().numpy()

        # normalize depth
        # depth_original = rearrange(points_original[..., 2], "h w -> () h w")
        depth_rendered = rearrange(points_rendered[..., 2], "h w layers -> layers h w")
        depth_gt = depth_rendered  # torch.cat([depth_original, depth_rendered], dim=0)
        depth_gt = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min())
        depth_gt = torch.clip(255 * depth_gt, 0, 255).to(torch.uint8).cpu().numpy()

        # outputs
        output_points = output_dict["points"][batch_idx]  # (h, w, samples, 3)
        output_colors = output_dict["colors"][batch_idx]  # (h, w, samples, 3)
        output_confs = output_dict["confs"][batch_idx]  # (h, w, samples, 1)

        # normalize depths
        depth_pred = rearrange(output_points[..., 2], "h w samples -> samples h w")
        depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min())
        depth_pred = torch.clip(255 * depth_pred, 0, 255).to(torch.uint8).cpu().numpy()

        # rearrange colors
        color_pred = rearrange(output_colors, "h w samples c -> samples h w c", c=3)
        color_pred = torch.clip(255 * color_pred, 0, 255).to(torch.uint8).cpu().numpy()

        # normalize confs
        conf_pred = rearrange(output_confs[..., 0], "h w samples -> samples h w")
        conf_pred = torch.sigmoid(conf_pred)
        conf_pred = torch.clip(255 * conf_pred, 0, 255).to(torch.uint8).cpu().numpy()

        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key=os.path.join(name, "image"),
                step=step,
                images=[img for img in image_all],
            )
            self.logger.log_image(
                key=os.path.join(name, "depth_gt"),
                step=step,
                images=[dep for dep in depth_gt],
            )
            self.logger.log_image(
                key=os.path.join(name, "depth_pred"),
                step=step,
                images=[dep for dep in depth_pred],
            )
            self.logger.log_image(
                key=os.path.join(name, "color_pred"),
                step=step,
                images=[color for color in color_pred],
            )
            self.logger.log_image(
                key=os.path.join(name, "conf_pred"),
                step=step,
                images=[conf for conf in conf_pred],
            )

        else:
            raise NotImplementedError(f"{self.logger} unsupported for logging images")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        backbone_lr = self.hparams.optimizer.keywords.pop("backbone_lr")
        head_lr = self.hparams.optimizer.keywords.pop("head_lr")
        optimizer = self.hparams.optimizer(
            params=[
                {"params": self.net.backbone.parameters(), "lr": backbone_lr},
                {"params": self.net.head.parameters(), "lr": head_lr},
            ]
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MeshHoGeModule(None, None, None, None)
