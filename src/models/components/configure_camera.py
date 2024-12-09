import numpy as np
import torch
import torch.nn.functional as F

# from cuml.cluster.hdbscan import HDBSCAN
from hdbscan import HDBSCAN
from jaxtyping import Float
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation


class ConfigureCamera(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.camera_immature_warning_flag = True

        # get_pose attributes
        self.clustering_imgsize = 64
        self.cluster_process_size = (self.clustering_imgsize, self.clustering_imgsize)
        min_cluster_size = int(self.clustering_imgsize * self.clustering_imgsize * 0.005)
        self.clusterer = HDBSCAN(min_cluster_size=min_cluster_size)

    def get_pose_simple(
        self, points: Float[torch.Tensor, "b h w 3"]
    ) -> Float[torch.Tensor, "b 4 4"]:
        angle = np.random.uniform(-np.pi / 16, np.pi / 16)
        inv_sign = -1 if angle > 0 else 1
        rot = Rotation.from_rotvec(angle * np.array([0, 1, 0])).as_matrix()
        rot = torch.tensor(rot, dtype=torch.float, device=points.device)

        trans = torch.tensor(
            [inv_sign * np.random.uniform(0, 0.2), 0.0, 0.0],
            dtype=torch.float,
            device=points.device,
        )

        w2c = (
            torch.eye(4, device=points.device).unsqueeze(0).expand(points.shape[0], -1, -1).clone()
        )
        w2c[:, :3, :3] = rot
        w2c[:, :3, 3] = trans
        return w2c

    def get_pose(self, points: Float[torch.Tensor, "b h w 3"]) -> Float[torch.Tensor, "b 4 4"]:
        """Find the closest object, and move closer to it but don't rotate the camera.

        By doing so, out-of-FoV regions never appear in novel views.
        """
        batch, height, width, channel = points.shape
        assert channel == 3

        point_cloud = (
            F.interpolate(points.permute(0, 3, 1, 2), self.cluster_process_size)
            .permute(0, 2, 3, 1)
            .reshape(batch, -1, 3)
        )
        point_cloud = (point_cloud - point_cloud.mean(axis=1, keepdims=True)) / (
            1e-3 + point_cloud.std(axis=1, keepdims=True)
        )

        w2c = torch.eye(4, device=points.device).unsqueeze(0).expand(batch, -1, -1).clone()

        for b in range(batch):
            pcd = point_cloud[b]
            cluster_labels = torch.as_tensor(
                self.clusterer.fit_predict(pcd.cpu().numpy()), device=points.device
            )

            # target camera possible area
            dep_max = pcd[..., 2].max()
            dep = torch.where(cluster_labels == -1, dep_max, pcd[..., 2])
            dep_min_idx = torch.argmin(dep)
            dep_min_label = cluster_labels[dep_min_idx]

            # uniform-randomly pick a point in the mask as camera destination
            cand_idx = torch.arange(len(cluster_labels), device=points.device)
            cand_idx = cand_idx[cluster_labels == dep_min_label]
            tgt_idx = cand_idx[
                torch.multinomial(torch.ones_like(cand_idx, dtype=torch.float32), 1)
            ]
            tgt_i, tgt_j = tgt_idx // self.clustering_imgsize, tgt_idx % self.clustering_imgsize
            tgt_i = tgt_i * height // self.clustering_imgsize
            tgt_j = tgt_j * width // self.clustering_imgsize
            tgt_xyz = points[b][tgt_i, tgt_j, :]
            # print(f"{tgt_i=}, {tgt_j=}, {tgt_xyz=}")

            trans_ratio = 0.1 + 0.4 * torch.rand(1, device=points.device)
            trans = (
                trans_ratio * tgt_xyz * (-1)
            )  # Since we want w2c, the translation must be inverted

            angle = 0  # np.random.uniform(-np.pi / 16, np.pi / 16)
            rot = Rotation.from_rotvec(angle * np.array([0, 1, 0])).as_matrix()
            rot = torch.tensor(rot, dtype=torch.float, device=points.device)

            w2c[b, :3, :3] = rot
            w2c[b, :3, 3] = trans

        return w2c

    def forward(
        self,
        intrinsics: Float[torch.Tensor, "b 3 3"],
        image: Float[torch.Tensor, "b h w c"],
        points: Float[torch.Tensor, "b h w 3"],
    ) -> PerspectiveCameras:
        if self.camera_immature_warning_flag:
            print("[ConfigureCamera] WARNING: camera configuration scheme is still immature!!!")
            self.camera_immature_warning_flag = False

        batch, height, width, channel = image.shape
        assert channel == 3

        pose = self.get_pose(points)

        assert intrinsics.shape == (batch, 3, 3)
        assert pose.shape == (batch, 4, 4)

        # image_size_list = [texture.shape[:2] for texture in meshes.textures.maps_list()]
        image_size_list = [(height, width) for _ in range(batch)]

        # Define a camera (OpenCV -> PyTorch3D)
        xy_flipvec = torch.tensor([-1, -1, 1], dtype=pose.dtype, device=pose.device)
        xy_flipmat_3x3 = torch.diag(xy_flipvec).unsqueeze(0)
        # xy_flipmat_4x4 = torch.tensor(
        #     [[[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
        #     dtype=pose.dtype,
        #     device=pose.device,
        # )

        # c2w = torch.linalg.inv(pose) @ xy_flipmat_4x4
        # w2c = torch.linalg.inv(c2w)
        # R = w2c[:, :3, :3].permute(0, 2, 1)
        # T = w2c[:, :3, 3]

        R = (xy_flipmat_3x3 @ pose[:, :3, :3]).permute(0, 2, 1)
        T = xy_flipvec * pose[:, :3, 3]

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
