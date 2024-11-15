import numpy as np
import torch
from jaxtyping import Float
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation


class ConfigureCamera(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.camera_immature_warning_flag = True

    def get_pose(self, points: Float[torch.Tensor, "b h w 3"]) -> Float[torch.Tensor, "b 4 4"]:
        angle = -2 * np.pi / 32
        rot = Rotation.from_rotvec(angle * np.array([0, 1, 0])).as_matrix()
        rot = torch.tensor(rot, dtype=torch.float, device=points.device)

        trans = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float, device=points.device)

        w2c = (
            torch.eye(4, device=points.device).unsqueeze(0).expand(points.shape[0], -1, -1).clone()
        )
        w2c[:, :3, :3] = rot
        w2c[:, :3, 3] = trans
        return w2c

    def forward(
        self,
        intrinsics: Float[torch.Tensor, "b 3 3"],
        image: Float[torch.Tensor, "b h w c"],
        points: Float[torch.Tensor, "b h w 3"],
    ) -> PerspectiveCameras:
        if self.camera_immature_warning_flag:
            print("[ConfigureCamera] WARNING: camera configuration scheme is terribly immature!!!")
            self.camera_immature_warning_flag = False

        batch, height, width, channel = image.shape
        assert channel == 3

        pose = self.get_pose(points)

        assert intrinsics.shape == (batch, 3, 3)
        assert pose.shape == (batch, 4, 4)

        # image_size_list = [texture.shape[:2] for texture in meshes.textures.maps_list()]
        image_size_list = [(height, width) for _ in range(batch)]

        # Define a camera (OpenCV -> PyTorch3D)
        xy_flipmat_4x4 = torch.tensor(
            [[[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
            dtype=torch.float,
            device="cuda",
        )

        c2w = torch.linalg.inv(pose) @ xy_flipmat_4x4
        w2c = torch.linalg.inv(c2w)
        R = w2c[:, :3, :3].permute(0, 2, 1)
        T = w2c[:, :3, 3]

        cameras = PerspectiveCameras(
            focal_length=intrinsics.diagonal(dim1=1, dim2=2)[:, :2],
            principal_point=intrinsics[:, :2, 2],
            R=R,
            T=T,
            device=image.device,
            in_ndc=False,
            image_size=image_size_list,
        )

        return cameras
