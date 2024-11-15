from typing import Optional

import torch
import torch.nn.functional as F
import utils3d
from jaxtyping import Float
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes

from third_party.MoGe.moge.model import MoGeModel


class MoGeMesh(torch.nn.Module):
    def __init__(
        self,
        depth_edge_thresh: float = 0.05,  # higher than the original to circumvent 'all depth edge' phenomenon in outdoor scenes
        invalid_mesh_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        valid_alpha: float = 1.0,
        background_alpha: float = 1.0,  # NOTE: When processing with HoGe, sky region is a 'valid' region
        invalid_alpha: float = -1.0,
    ) -> None:
        super().__init__()
        assert 0 <= min(invalid_mesh_color) and max(invalid_mesh_color) <= 1
        assert len(invalid_mesh_color) == 3
        self.moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
        self.depth_edge_thresh = depth_edge_thresh
        self.invalid_mesh_color = invalid_mesh_color
        self.valid_alpha = valid_alpha
        self.invalid_alpha = invalid_alpha
        self.background_alpha = background_alpha

    def forward(
        self,
        image_ch_last: Float[torch.Tensor, "b h w c"],
    ) -> tuple[Float[torch.Tensor, "b 3 3"], Meshes, Float[torch.Tensor, "b h w 3"]]:
        image_ch_last = image_ch_last.clone()  # (B, H, W, 3)
        batch, height, width, channel = image_ch_last.shape
        dtype, device = image_ch_last.dtype, image_ch_last.device
        assert channel == 3, f"{image_ch_last.shape=}"

        # Infer
        output = self.moge.infer(image_ch_last.permute(0, 3, 1, 2))
        points = output["points"]  # (B, H, W, 3)
        depth = output["depth"]  # (B, H, W)
        mask = output["mask"]  # (B, H, W)
        intrinsics = output["intrinsics"]  # (B, 3, 3)

        # for novel rendering purpose, instantiate the sky region at depth=depth[mask].max()
        largest_depth, _ = torch.where(mask, depth, 0).reshape(batch, -1).max(dim=1)
        depth = torch.where(mask, depth, largest_depth.reshape(batch, 1, 1))
        points = utils3d.torch.unproject_cv(
            utils3d.torch.image_uv(
                width=width, height=height, dtype=points.dtype, device=points.device
            ),
            depth,
            extrinsics=None,
            intrinsics=intrinsics[..., None, :, :],
        )  # (B, H, W, 3)

        # change intrinsic to absolute coordinate values
        assert intrinsics.shape == (batch, 3, 3), f"{intrinsics.shape=}"
        K = torch.zeros_like(intrinsics)
        K[:, 0, :] = intrinsics[:, 0, :] * width
        K[:, 1, :] = intrinsics[:, 1, :] * height
        K[:, 2, 2] = 1

        # mark invalid mesh faces
        depth_edge = calc_depth_edge(depth, rtol=self.depth_edge_thresh, mask=mask)  # (B, H, W)
        image_ch_last[depth_edge, :] = torch.tensor(
            self.invalid_mesh_color, dtype=dtype, device=device
        ).reshape(1, 3)

        # construct meshes (visible=1, background=self.background_alpha, invalid=self.invalid_alpha)
        alphamap = torch.full(
            (batch, height, width, 1), fill_value=self.valid_alpha, dtype=dtype, device=device
        )
        alphamap[~mask, :] = self.background_alpha
        alphamap[depth_edge | (depth < 0), :] = self.invalid_alpha
        image_alpha_ch_last = torch.cat([image_ch_last, alphamap], dim=-1)

        faces_list = []
        verts_list = []
        vert_uv_list = []
        for b in range(batch):  # TODO: OPTIMIZE image_mesh()
            faces, vertices, vertex_colors, vertex_uvs = image_mesh(
                points[b],
                image_alpha_ch_last[b],
                utils3d.torch.image_uv(height=height, width=width, device=device),
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
            maps=image_alpha_ch_last,
            faces_uvs=faces_list,
            verts_uvs=vert_uv_list,
            align_corners=False,
        )

        # Initialise the mesh with textures
        meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex_list)

        return K, meshes, points


################################################################################################################################

# Modified from utils3d (https://github.com/EasternJournalist/utils3d/tree/main) for torch


def triangulate(
    faces: torch.Tensor,
    vertices: Optional[torch.Tensor] = None,
    backslash: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triangulate a polygonal mesh.

    Args:
        faces (torch.Tensor): [L, P] polygonal faces
        vertices (torch.Tensor, optional): [N, 3] 3-dimensional vertices.
            If given, the triangulation is performed according to the distance
            between vertices. Defaults to None.
        backslash (torch.Tensor, optional): [L] boolean array indicating
            how to triangulate the quad faces. Defaults to None.

    Returns:
        (np.ndarray): [L * (P - 2), 3] triangular faces
    """
    if faces.shape[-1] == 3:
        return faces
    P = faces.shape[-1]
    if vertices is not None:
        assert faces.shape[-1] == 4, "now only support quad mesh"
        if backslash is None:
            backslash = torch.linalg.norm(
                vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=-1
            ) < torch.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 3]], axis=-1)
    if backslash is None:
        loop_indice = torch.stack(
            [
                torch.zeros(P - 2, dtype=torch.int, device=faces.device),
                torch.arange(1, P - 1, 1, dtype=torch.int, device=faces.device),
                torch.arange(2, P, 1, dtype=torch.int, device=faces.device),
            ],
            dim=1,
        )
        return faces[:, loop_indice].reshape((-1, 3))
    else:
        assert faces.shape[-1] == 4, "now only support quad mesh"
        faces = torch.where(
            backslash[:, None], faces[:, [0, 1, 2, 0, 2, 3]], faces[:, [0, 1, 3, 3, 1, 2]]
        ).reshape((-1, 3))
        return faces


def remove_unreferenced_vertices(
    faces: torch.Tensor, *vertice_attrs, return_indices: bool = False
) -> tuple[torch.Tensor, ...]:
    """Remove unreferenced vertices of a mesh. Unreferenced vertices are removed, and the face
    indices are updated accordingly.

    Args:
        faces (torch.Tensor): [T, P] face indices
        *vertice_attrs: vertex attributes

    Returns:
        faces (torch.Tensor): [T, P] face indices
        *vertice_attrs: vertex attributes
        indices (torch.Tensor, optional): [N] indices of vertices that are kept. Defaults to None.
    """
    P = faces.shape[-1]
    fewer_indices, inv_map = torch.unique(faces, return_inverse=True)
    faces = inv_map.to(torch.int).reshape(-1, P)
    ret = [faces]
    for attr in vertice_attrs:
        ret.append(attr[fewer_indices])
    if return_indices:
        ret.append(fewer_indices)
    return tuple(ret)


def image_mesh(
    *image_attrs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tri: bool = False,
    return_indices: bool = False,
) -> tuple[torch.Tensor, ...]:
    """
    Get x quad mesh regarding image pixel uv coordinates as vertices and image grid as faces.
    NOTE: ASSUME CHANNEL_LAST INPUTS!!!!!!

    Args:
        *image_attrs (torch.Tensor): image attributes in shape (height, width, [channels])
        mask (torch.Tensor, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

    Returns:
        faces (torch.Tensor): faces connecting neighboring pixels. shape (T, 4) if tri is False, else (T, 3)
        *vertex_attrs (torch.Tensor): vertex attributes in corresponding order with input image_attrs
        indices (torch.Tensor, optional): indices of vertices in the original mesh
    """
    assert (len(image_attrs) > 0) or (
        mask is not None
    ), "At least one of image_attrs or mask should be provided"
    height, width = image_attrs[0].shape[:2] if mask is None else mask.shape
    assert all(
        img.shape[:2] == (height, width) for img in image_attrs
    ), "All image_attrs should have the same shape"

    device = image_attrs[0].device if len(image_attrs) > 0 else mask.device

    row_faces = torch.stack(
        [
            torch.arange(0, width - 1, dtype=torch.int, device=device),
            torch.arange(width, 2 * width - 1, dtype=torch.int, device=device),
            torch.arange(1 + width, 2 * width, dtype=torch.int, device=device),
            torch.arange(1, width, dtype=torch.int, device=device),
        ],
        dim=1,
    )
    faces = (
        torch.arange(0, (height - 1) * width, width, dtype=torch.int, device=device)[:, None, None]
        + row_faces[None, :, :]
    ).reshape((-1, 4))
    if mask is None:
        if tri:
            faces = triangulate(faces)
        ret = [faces, *(img.reshape(-1, *img.shape[2:]) for img in image_attrs)]
        if return_indices:
            ret.append(torch.arange(height * width, dtype=torch.int))
        return tuple(ret)
    else:
        quad_mask = (mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]).ravel()
        faces = faces[quad_mask]
        if tri:
            faces = triangulate(faces)
        return remove_unreferenced_vertices(
            faces,
            *(x.reshape(-1, *x.shape[2:]) for x in image_attrs),
            return_indices=return_indices,
        )


def calc_depth_edge(
    depth: torch.Tensor,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: torch.Tensor = None,
) -> torch.BoolTensor:
    """Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have
    a large difference in depth.

    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(
            -depth, kernel_size, stride=1, padding=kernel_size // 2
        )
    else:  # NOTE: Modified from the original (not -torch.inf but +torch.inf in the first term)
        diff = F.max_pool2d(
            torch.where(mask, depth, torch.inf), kernel_size, stride=1, padding=kernel_size // 2
        ) + F.max_pool2d(
            torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2
        )

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge


if __name__ == "__main__":
    _ = MoGeMesh()
