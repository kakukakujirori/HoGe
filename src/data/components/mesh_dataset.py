import glob
import os
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
import utils3d
from jaxtyping import Float
from PIL import Image
from pycocotools.coco import COCO
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes, join_meshes_as_scene
from torch.utils.data import Dataset


class MeshDataset(Dataset):
    def __init__(
        self,
        original_image_dir: str,
        composite_image_dir: str,
        annot_json: str,
        longer_image_edge_size: int = 512,
        augmentation_list: list = [],
    ) -> None:
        assert os.path.isdir(original_image_dir)
        assert os.path.isdir(
            os.path.join(composite_image_dir, "image")
        ), f"Not found: {os.path.join(composite_image_dir, 'image')}"
        assert os.path.isdir(
            os.path.join(composite_image_dir, "depth")
        ), f"Not found: {os.path.join(composite_image_dir, 'depth')}"
        assert os.path.isfile(annot_json)
        self.coco = COCO(annot_json)
        self.image_ids = self.coco.getImgIds()
        self.original_image_dir = original_image_dir
        self.composite_image_dir = composite_image_dir
        self.longer_image_edge_size = longer_image_edge_size
        self.aug = A.Compose(augmentation_list)

    @staticmethod
    def unproject(
        depth: Float[np.ndarray, "h w"],
        f_px: float,
        height: int,
        width: int,
        sy: int,
        sx: int,
        ty: int,
        tx: int,
    ) -> Float[np.ndarray, "h w 3"]:
        y, x = np.meshgrid(
            np.arange(sy, ty, dtype=depth.dtype),
            np.arange(sx, tx, dtype=depth.dtype),
            indexing="ij",
        )
        x = (x - width / 2) / f_px
        y = (y - height / 2) / f_px
        ret = np.stack([x, y, np.ones_like(depth)], axis=-1)
        return np.where(
            np.isfinite(depth)[..., None],
            ret * depth[..., None],
            np.full_like(ret, np.inf),
        )

    def ann_to_pcd(
        self, ann: dict, f_len: float, label_offset: int = 1000, return_mask: bool = False
    ):
        if ann["id"] % label_offset == 0:
            return [None] * (3 if return_mask else 2)
        img_data = self.coco.loadImgs(ann["image_id"])[0]
        sx, sy, w, h = ann["bbox"]

        obj_color = cv2.cvtColor(
            cv2.imread(os.path.join(self.composite_image_dir, f"image/{ann['id']}.jpg")),
            cv2.COLOR_BGR2RGB,
        )

        obj_depth = np.load(os.path.join(self.composite_image_dir, f"depth/{ann['id']}.npy"))
        obj_pts = self.unproject(
            obj_depth,
            f_len,
            img_data["height"],
            img_data["width"],
            int(sy),
            int(sx),
            int(sy + h),
            int(sx + w),
        )

        mask = self.coco.annToMask(ann)[
            int(sy) : int(sy + obj_pts.shape[0]), int(sx) : int(sx + obj_pts.shape[1])
        ]
        if return_mask:
            return obj_pts, obj_color / 255, mask
        else:
            return obj_pts[mask > 0.5], obj_color[mask > 0.5] / 255

    def meshify(self, img_id: int, f_len: Optional[float]) -> Meshes:
        image_info = self.coco.loadImgs(img_id)[0]
        image_file_name = image_info["file_name"]
        f_len = f_len or image_info["focal_length"]
        image = cv2.cvtColor(
            cv2.imread(os.path.join(self.original_image_dir, image_file_name)), cv2.COLOR_BGR2RGB
        )
        depth = np.load(
            os.path.join(
                self.composite_image_dir, "depth", image_file_name.replace(".jpg", ".npy")
            )
        )
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        assert image.shape[:2] == depth.shape

        points_list = []
        colors_list = []
        masks_list = []

        bkg_pts = self.unproject(
            depth, f_len, image.shape[0], image.shape[1], 0, 0, image.shape[0], image.shape[1]
        )
        bkg_colors = image / 255.0
        bkg_mask = np.isfinite(depth)  # identify sky region

        # bkg_depth = depth.copy()
        # bkg_normal, bkg_normal_mask = utils3d.numpy.points_to_normals(bkg_pts, mask=bkg_mask)
        # bkg_mask = bkg_mask & ~(
        #     utils3d.numpy.depth_edge(bkg_depth, rtol=0.03, mask=bkg_mask)
        #     & utils3d.numpy.normals_edge(bkg_normal, tol=5, mask=bkg_normal_mask)
        # )

        points_list.append(bkg_pts)
        colors_list.append(bkg_colors)
        masks_list.append(bkg_mask)

        for ann in anns:
            obj_pts, obj_colors, obj_mask = self.ann_to_pcd(ann, f_len, return_mask=True)
            if obj_pts is None and obj_colors is None:
                continue

            # obj_depth = obj_pts[:, :, 2]
            # obj_normal, obj_normal_mask = utils3d.numpy.points_to_normals(obj_pts, mask=obj_mask)
            # obj_mask = obj_mask & ~(
            #     utils3d.numpy.depth_edge(obj_depth, rtol=0.03, mask=obj_mask)
            #     & utils3d.numpy.normals_edge(obj_normal, tol=5, mask=obj_normal_mask)
            # )

            points_list.append(obj_pts)
            colors_list.append(obj_colors)
            masks_list.append(obj_mask)

        # convert to mesh components
        faces_list = []
        verts_list = []
        verts_uv_list = []
        for pts, colors, mask in zip(points_list, colors_list, masks_list):
            height, width = mask.shape
            assert pts.shape == (height, width, 3)
            assert colors.shape == (height, width, 3)

            # some data is saved as uint8
            mask = mask.astype(bool)

            faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                pts,
                colors,
                utils3d.numpy.image_uv(width=width, height=height),
                mask=mask,
                tri=True,
            )
            vertex_uvs[:, 1] = 1 - vertex_uvs[:, 1]
            faces_list.append(torch.from_numpy(faces))
            verts_list.append(torch.from_numpy(vertices))
            verts_uv_list.append(torch.from_numpy(vertex_uvs).float())

        colors_list = [torch.from_numpy(colors).float() for colors in colors_list]

        # Create a textures object
        tex_list = TexturesUV(
            maps=colors_list,
            faces_uvs=faces_list,
            verts_uvs=verts_uv_list,
            align_corners=False,
        )

        # Initialise the mesh with textures
        meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex_list)
        meshes = join_meshes_as_scene(meshes)
        return meshes

    def __getitem__(self, index: int) -> Meshes:
        # load an image
        img_id = self.image_ids[index]
        image_file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        img = cv2.imread(os.path.join(self.composite_image_dir, "image", image_file_name))
        if img is None:
            raise FileNotFoundError(
                f"Image not found: {os.path.join(self.composite_image_dir, 'image', image_file_name)}"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # augmentations
        img = self.aug(image=img)["image"]

        # resize (with aspect ratio preserved) and zero padding
        scale_factor = self.longer_image_edge_size / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        if img.shape[0] == self.longer_image_edge_size:
            top, bottom = 0, 0
            left = (self.longer_image_edge_size - img.shape[1]) // 2
            right = self.longer_image_edge_size - img.shape[1] - left
        elif img.shape[1] == self.longer_image_edge_size:
            left, right = 0, 0
            top = (self.longer_image_edge_size - img.shape[0]) // 2
            bottom = self.longer_image_edge_size - img.shape[0] - top
        else:
            raise RuntimeError(
                f"This should not happen: {img.shape=}, {self.longer_image_edge_size=}"
            )

        # padding (NOTE: mask must come first!!!)
        mask = cv2.copyMakeBorder(
            np.ones_like(img[:, :, 0]), top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

        # convert to tensors (NOTE: CHANNEL LAST!!!)
        img_t = torch.from_numpy(img).float() / 255
        mask_t = torch.from_numpy(mask).float()

        # construct GT mesh
        f_len = 1000.0  # TODO: ATTACH F_LEN IN JSON!!!
        mesh = self.meshify(img_id, f_len)

        return {
            "composite_image": img_t,
            "valid_mask": mask_t,
            "meshes": mesh,
            "focal_len": torch.tensor([f_len], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.image_ids)
