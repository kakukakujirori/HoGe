from typing import Optional
import torch
import torch.nn.functional as F

from einops import rearrange
from jaxtyping import Float
from pytorch3d.ops.knn import knn_points


class OptimalAlignmentLoss(torch.nn.Module):
    def __init__(
            self,
            visible_weight: float = 1.0,
            occlusion_weight: float = 1.0,
            dist_weight: float = 1.0,
            color_weight: float = 1.0,
            conf_weight: float = 1.0,
        ) -> None:
        super().__init__()
        self.visible_weight = visible_weight
        self.occlusion_weight = occlusion_weight
        self.dist_weight = dist_weight
        self.color_weight = color_weight
        self.conf_weight = conf_weight

        self.optimal_alignment_solver_unimplemented_warning_flag = True

    def forward(
            self,
            input_dict: dict[str, torch.Tensor],
            output_dict: dict[str, torch.Tensor],
        ) -> torch.Tensor:
        # gather valid points pixelwisely (maybe redundant, but to make things clear)
        input_points_gathered, input_texels_gathered = self.gather_valid_points(input_dict)  # (b, h, w, layers, 3)
        input_masks = input_dict.get("masks_rendered", None)  # (b, h, w)
        output_points = output_dict["points"]  # (b, h, w, layers, 3)

        batch, height, width, layers, channel = input_points_gathered.shape
        assert channel == 3
        assert input_points_gathered.shape == output_points.shape, f"{input_points_gathered.shape=}, {output_points.shape=}"
        assert input_masks is None or (batch, height, width) == input_masks.shape, f"{input_masks.shape=}"

        # optimal scale and shift are determined from the first layer results
        input_points_first = input_points_gathered[:, :, :, 0, :]
        target_points_first = output_points[:, :, :, 0, :]
        scale, shift = self.optimal_alignment_solver(input_points_first, target_points_first, input_masks)
        output_points_aligned = scale.reshape(batch, 1, 1, 1, 1) * output_points + shift.reshape(batch, 1, 1, 1, 3)

        # first (visible) layer loss calculation
        loss_dict_visible = self.visible_region_loss(
            input_points_gathered[:, :, :, 0, :],
            input_texels_gathered[:, :, :, 0, :],
            output_points_aligned[:, :, :, 0, :],
            output_dict["colors"][:, :, :, 0, :],
            output_dict["confs"][:, :, :, 0, :],
        )

        # allotment of second/third points -> loss calculation
        loss_dict_occluded = self.occluded_region_loss(
            input_points_gathered[:, :, :, 1:, :],
            input_texels_gathered[:, :, :, 1:, :],
            output_points_aligned[:, :, :, 1:, :],
            output_dict["colors"][:, :, :, 1:, :],
            output_dict["confs"][:, :, :, 1:, :],
        )

        # apply weighting to each loss type
        loss_dict_visible["dist_loss"] *= self.dist_weight
        loss_dict_visible["color_loss"] *= self.color_weight
        loss_dict_visible["conf_loss"] *= self.conf_weight

        loss_dict_occluded["dist_loss"] *= self.dist_weight
        loss_dict_occluded["color_loss"] *= self.color_weight
        loss_dict_occluded["conf_loss"] *= self.conf_weight

        # apply weighting to visible/occluded
        loss_dict = {}
        for key, val in loss_dict_visible.items():
            loss_dict["visible_" + key] = val * self.visible_weight
        for key, val in loss_dict_occluded.items():
            loss_dict["occluded_" + key] = val * self.occlusion_weight

        return loss_dict

    @staticmethod
    def gather_valid_points(input_dict: dict[str, torch.Tensor]):
        points: Float[torch.Tensor, "b h w layers 3"] = input_dict["points_rendered"]
        texels: Float[torch.Tensor, "b h w layers 4"] = input_dict["texels_rendered"]
        valid_masks = texels[:, :, :, :, 3:4] > 0.9  # NOTE: sky region is invalid here
        points_filled = torch.where(valid_masks, points, torch.inf)
        points_filled_z = points_filled[..., 2:3]
        points_filled_z_sorted, sort_idx = torch.sort(points_filled_z, dim=-2)
        points_sorted = torch.gather(points, dim=-2, index=sort_idx.expand(-1, -1, -1, -1, 3))
        texels_sorted = torch.gather(texels, dim=-2, index=sort_idx.expand(-1, -1, -1, -1, 4))
        return points_sorted, texels_sorted

    @staticmethod
    @torch.no_grad()
    def optimal_alignment_solver(
            input_points: Float[torch.Tensor, "b h w 3"],
            output_points: Float[torch.Tensor, "b h w 3"],
            valid_masks: Float[torch.Tensor, "b h w"],
            process_size: int = 64,
            mask_sparsify_ratio: float = 0.99,
            outlier_thresh: float = 10,
            coerce_positive_scale: bool = False,
        ):
        """
        Deduce the optimal scale and shift (s, t) that minimizes \sum |input_points - (s * output_points + t)|_1 / input_points[..., 2]
        """
        eps = 1e-6

        batch, height, width, channel = input_points.shape
        assert channel == 3
        assert output_points.shape == (batch, height, width, channel)
        assert valid_masks.shape == (batch, height, width)
        input_points = rearrange(F.interpolate(rearrange(input_points, "b h w c -> b c h w", c=3), (process_size, process_size)), "b c h w -> b (h w) c", c=3)
        output_points = rearrange(F.interpolate(rearrange(output_points, "b h w c -> b c h w", c=3), (process_size, process_size)), "b c h w -> b (h w) c", c=3)
        valid_masks = rearrange(F.interpolate(rearrange(valid_masks.float(), "b h w -> b () h w"), (process_size, process_size)), "b () h w -> b (h w)").bool()

        # sparsify the mask for faster computation
        valid_masks = valid_masks & (torch.rand_like(valid_masks.float()) > mask_sparsify_ratio)

        # maybe batchfiable by setting the ~mask value torch.inf??????
        ret_scale_list = []
        ret_shift_list = []
        for b in range(batch):

            input_pts = input_points[b]
            output_pts = output_points[b]
            mask = valid_masks[b]

            # want to solve min_{s, t} |s * Q + (0,0,t) - P|
            P = input_pts[mask]  # (N, 3)
            Q = output_pts[mask]  # (N, 3)
            N, _ = P.shape

            # print(f"{P=}\n{Q=}")

            # remove t: min_s |s * Q2 - P2| (t = P_k - s * Q_k)
            Pz = P[:, 2:3]  # (N, 1)
            Pzk_Pzi = Pz.unsqueeze(0) - Pz.unsqueeze(1)  # (1, N, 1) - (N, 1, 1) -> (Nk, Ni, 1) // Pz_Pz[k, i, :] = Pz[i, :] - Pz[k, :]
            Pxy_Pxy = P[:, :2].unsqueeze(0).expand(N, -1, -1)
            P2 = torch.cat([Pxy_Pxy, Pzk_Pzi], dim=-1)  # (N, N, 3) // P2[k, i, :] = [Px^i, Py^i, Pz^i - Pz^k]

            Qz = Q[:, 2:3]  # (N, 1)
            Qzk_Qzi = Qz.unsqueeze(0) - Qz.unsqueeze(1)  # (1, N, 1) - (N, 1, 1) -> (Nk, Ni, 1) // Qz_Qz[k, i, :] = Qz[i, :] - Qz[k, :]
            Qxy_Qxy = Q[:, :2].unsqueeze(0).expand(N, -1, -1)
            Q2 = torch.cat([Qxy_Qxy, Qzk_Qzi], dim=-1)  # (N, N, 3) // Q2[k, i, :] = [Qx^i, Qy^i, Qz^i - Qz^k]

            # make all Q2 elements positive (zero => remove by setting P=inf, negative => invert)
            P2 = torch.where(Q2 > 0, P2, -P2)
            Q2 = torch.where(Q2 > 0, Q2, -Q2)
            Q2_zero_mask = torch.abs(Q2) < eps
            P2[Q2_zero_mask] = torch.inf
            Q2[Q2_zero_mask] = eps

            # print(f"{P2=}\n{Q2=}")

            weight = (1 / (torch.abs(Pz) + eps)).reshape(1, N, 1)
            values = (weight * Q2).reshape(N, N * 3)
            assert torch.all(weight > 0), f"{weight=}"
            assert torch.all(Q2 > 0), f"{Q2=}"
            # print(f"{values=}")
            ratio = (P2 / (Q2 + eps)).reshape(N, N * 3)
            ratio_minus = ((weight * P2 - outlier_thresh) / (weight * Q2 + eps)).reshape(N, N * 3)
            ratio_plus = ((weight * P2 + outlier_thresh) / (weight * Q2 + eps)).reshape(N, N * 3)

            ratio_sorted, ratio_idx = torch.sort(ratio, dim=-1)
            ratio_minus_sorted, ratio_minus_idx = torch.sort(ratio_minus, dim=-1)
            ratio_plus_sorted, ratio_plus_idx = torch.sort(ratio_plus, dim=-1)

            # print(f"{ratio=}\n{ratio_sorted=}\n{ratio_idx=}")
            del P2, Q2, weight, ratio_minus, ratio_plus

            ratio_ordered_cumsum = torch.cumsum(torch.gather(values, -1, ratio_idx), dim=-1)
            ratio_minus_ordered_cumsum = torch.cumsum(torch.gather(values, -1, ratio_minus_idx), dim=-1)
            ratio_plus_ordered_cumsum = torch.cumsum(torch.gather(values, -1, ratio_plus_idx), dim=-1)

            # add sentinels for binary search
            zero_padding = torch.tensor([[0]], dtype=values.dtype, device=values.device).expand(N, 1)
            inf_padding = torch.tensor([[torch.inf]], dtype=values.dtype, device=values.device).expand(N, 1)
            ratio_ordered_cumsum = torch.cat([zero_padding, ratio_ordered_cumsum, inf_padding], dim=-1)
            ratio_minus_ordered_cumsum = torch.cat([zero_padding, ratio_minus_ordered_cumsum, inf_padding], dim=-1)
            ratio_plus_ordered_cumsum = torch.cat([zero_padding, ratio_plus_ordered_cumsum, inf_padding], dim=-1)

            # print(f"{ratio_ordered_cumsum=}")

            # ratio = (scale candidates) by Lemma 1
            left_idx = torch.searchsorted(ratio_sorted, ratio, right=False)
            left_minus_idx = torch.searchsorted(ratio_minus_sorted, ratio, right=False)
            left_plus_idx = torch.searchsorted(ratio_plus_sorted, ratio, right=False)
            left_grad = 2 * torch.gather(ratio_ordered_cumsum, -1, left_idx) \
                        - torch.gather(ratio_minus_ordered_cumsum, -1, left_minus_idx) \
                        - torch.gather(ratio_plus_ordered_cumsum, -1, left_plus_idx)  # (N, N * 3)

            # print(f"{left_idx=}\n{torch.gather(ratio_ordered_cumsum, -1, left_idx)=}", flush=True)

            right_idx = torch.searchsorted(ratio_sorted, ratio, right=True)
            right_minus_idx = torch.searchsorted(ratio_minus_sorted, ratio, right=True)
            right_plus_idx = torch.searchsorted(ratio_plus_sorted, ratio, right=True)
            right_grad = 2 * torch.gather(ratio_ordered_cumsum, -1, right_idx) \
                        - torch.gather(ratio_minus_ordered_cumsum, -1, right_minus_idx) \
                        - torch.gather(ratio_plus_ordered_cumsum, -1, right_plus_idx)  # (N, N * 3)

            # print(f"{right_idx=}\n{torch.gather(ratio_ordered_cumsum, -1, right_idx)=}", flush=True)

            # extract scale candidates that satisfy left_grad(s) < 0 <= right_grad
            extrema_mask = torch.logical_and(left_grad < -eps, eps <= right_grad) & ratio.isfinite() # (N, N * 3)
            if coerce_positive_scale:
                extrema_mask = extrema_mask & (ratio > 0)
            extrema_idx = extrema_mask.nonzero()  # (M, 2)

            if len(extrema_idx) > 0:
                scale_cand = ratio[extrema_mask]  # (M,)
                k_cand = extrema_idx[:, 0]  # (M,)
                shift_cand = torch.zeros((k_cand.shape[0], 3), dtype=Pz.dtype, device=Pz.device)
                shift_cand[:, 2] = Pz[k_cand].flatten() - scale_cand * Qz[k_cand].flatten()  # (M,)
                cost = torch.sum(torch.abs(P - scale_cand.reshape(-1, 1, 1) * Q - shift_cand.reshape(-1, 1, 3)) / torch.abs(Pz), dim=[-2,-1])  # (M, 3) -> (M,)
                cost_min_idx = torch.argmin(cost)

                # print(f"{left_grad=}\n{right_grad=}\n{extrema_mask=}\n{extrema_idx=}\n{scale_cand=}\n{shift_cand=}\n{cost_min_idx=}")

                ret_scale = scale_cand[cost_min_idx]
                ret_shift = shift_cand[cost_min_idx]

            else:
                print(f"[optimal_alignment_solver] No solution found. The mask may be empty:\n{torch.sum(mask)=}\n{Pz=}\n{Qz=}")
                try:
                    shift_cand = (Pz - Qz).reshape(1, N)
                    assert Pz.shape == Qz.shape == (N, 1) and shift_cand.shape == (1, N)
                    cost = torch.sum(torch.abs(Pz - Qz - shift_cand) / torch.abs(Pz), dim=0)
                    cost_min_idx = torch.argmin(cost)
                    ret_shift = shift_cand[cost_min_idx]
                    ret_scale = torch.ones(1, dtype=ret_shift.dtype, device=ret_shift.device)[0]
                except IndexError as e:
                    print(e)
                    ret_shift = torch.zeros(3, dtype=input_points.dtype, device=input_points.device)
                    ret_scale = torch.ones(1, dtype=ret_shift.dtype, device=ret_shift.device)[0]

            ret_scale_list.append(ret_scale)
            ret_shift_list.append(ret_shift)

        ret_scale_list = torch.stack(ret_scale_list, dim=0)
        ret_shift_list = torch.stack(ret_shift_list, dim=0)
        return ret_scale_list, ret_shift_list

    def inv_depth_weighted_l1(
            self,
            input_points: Float[torch.Tensor, "... 3"],
            output_points: Float[torch.Tensor, "... 3"],
            masks: Optional[Float[torch.Tensor, "..."]] = None,
        ):
        assert input_points.shape == output_points.shape
        assert input_points.shape[-1] == 3

        if masks is not None:
            assert masks.dtype == torch.bool
            if torch.any(masks):
                masks = masks.squeeze(-1)
                diff = torch.abs(input_points[masks, :] - output_points[masks, :])
                loss_per_pixel = diff / input_points[masks][..., 2:3]  # weighting is inv-proportional to depth
            else:
                loss_per_pixel = 0 * torch.mean(output_points)  # dummy
        else:
            diff = torch.abs(input_points - output_points)
            loss_per_pixel = diff / input_points[..., 2:3]

        return torch.mean(loss_per_pixel)

    def visible_region_loss(
            self,
            input_points: Float[torch.Tensor, "b h w 3"],
            input_texels: Float[torch.Tensor, "b h w 4"],
            output_points: Float[torch.Tensor, "b h w 3"],
            output_colors: Float[torch.Tensor, "b h w 3"],
            output_confs: Float[torch.Tensor, "b h w 1"],
        ):
        batch, height, width, ch = input_points.shape
        assert ch == 3
        assert input_points.shape == (batch, height, width, 3)
        assert input_texels.shape == (batch, height, width, 4)
        assert output_points.shape == (batch, height, width, 3)
        assert output_colors.shape == (batch, height, width, 3)
        assert output_confs.shape == (batch, height, width, 1)

        input_colors = input_texels[..., :3]
        input_masks = (input_texels[..., 3] > 0.9)  # NOTE: sky is invalid

        if torch.any(input_masks):
            dist_loss = self.inv_depth_weighted_l1(
                input_points=input_points,
                output_points=output_points,
                masks=input_masks,
            )
            color_loss = F.mse_loss(
                input_colors[input_masks],
                output_colors[input_masks],
            )
            conf_loss = F.binary_cross_entropy_with_logits(
                output_confs,
                input_masks.reshape_as(output_confs).float(),
            )
        else:
            dist_loss = 0 * torch.mean(output_points)
            color_loss = 0 * torch.mean(output_colors)
            conf_loss = 0 * torch.mean(output_confs)

        total_loss = {
            "dist_loss": dist_loss,
            "color_loss": color_loss,
            "conf_loss": conf_loss,
        }

        return total_loss

    def occluded_region_loss(
            self,
            input_points_occ: Float[torch.Tensor, "b h w occ 3"],
            input_texels_occ: Float[torch.Tensor, "b h w occ 4"],
            output_points_occ: Float[torch.Tensor, "b h w occ 3"],
            output_colors_occ: Float[torch.Tensor, "b h w occ 3"],
            output_confs_occ: Float[torch.Tensor, "b h w occ 1"],
        ):
        batch, height, width, occ, ch = input_points_occ.shape
        assert ch == 3
        assert input_points_occ.shape == (batch, height, width, occ, 3)
        assert input_texels_occ.shape == (batch, height, width, occ, 4)
        assert output_points_occ.shape == (batch, height, width, occ, 3)
        assert output_colors_occ.shape == (batch, height, width, occ, 3)
        assert output_confs_occ.shape == (batch, height, width, occ, 1)

        # handle per pixel
        input_points_flatten = input_points_occ.reshape(-1, occ, 3)
        input_colors_flatten = input_texels_occ[..., :3].reshape(-1, occ, 3)
        input_masks_flatten = (input_texels_occ[..., 3:].reshape(-1, occ, 1) > 0.9)  # NOTE: sky is invalid
        output_points_flatten = output_points_occ.reshape(-1, occ, 3)
        output_colors_flatten = output_colors_occ.reshape(-1, occ, 3)
        output_confs_flatten = output_confs_occ.reshape(-1, occ, 1)

        # apply losses only where GT exists, so nearest search from the GT-side
        input_points_len = input_masks_flatten.sum(dim=1).flatten()  # NOTE: input_points must be gathered beforehand!!!
        assert input_points_len.shape == (batch * height * width,)
        _dist, idx, _ = knn_points(
            p1=input_points_flatten,  # (b * h * w, occ, 3)
            p2=output_points_flatten,  # (b * h * w, occ, 3)
            lengths1=input_points_len,  # (b * h * w,)
            lengths2=None,
        )  # idx.shape = (b * h * w, occ, 1)
        output_points_fetched = torch.gather(output_points_flatten, dim=1, index=idx.expand(-1, -1, 3))  # (b * h * w, occ, 3)
        output_colors_fetched = torch.gather(output_colors_flatten, dim=1, index=idx.expand(-1, -1, 3))  # (b * h * w, occ, 3)
        output_confs_fetched = torch.gather(output_confs_flatten, dim=1, index=idx.expand(-1, -1, 1))  # (b * h * w, occ, 1)

        if occ == 2:
            # If two GT points exist, assign each GT point to each pred points
            two_points_mask = (input_points_len == 2)
            if torch.any(two_points_mask):
                two_points_dist_loss = self.inv_depth_weighted_l1(
                    input_points=input_points_flatten,
                    output_points=output_points_flatten,
                    masks=two_points_mask,
                )
                two_points_color_loss = F.mse_loss(
                    input_colors_flatten[two_points_mask],
                    output_colors_flatten[two_points_mask]
                )
                two_points_conf_loss = F.binary_cross_entropy_with_logits(
                    output_confs_flatten[two_points_mask],
                    input_masks_flatten[two_points_mask].float(),
                )
            else:
                two_points_dist_loss = two_points_color_loss = two_points_conf_loss = 0 * torch.mean(output_points_occ)
            # If only one GT point exist, apply the loss to the single prediction counterpart
            one_point_mask = (input_points_len == 1)
            if torch.any(one_point_mask):
                one_point_dist_loss = self.inv_depth_weighted_l1(
                    input_points=input_points_flatten[:, 0:1, :],
                    output_points=output_points_fetched[:, 0:1, :],
                    masks=one_point_mask,
                )
                one_point_color_loss = F.mse_loss(
                    input_colors_flatten[one_point_mask][:, 0:1, :],
                    output_colors_fetched[one_point_mask][:, 0:1, :],
                )
                one_point_conf_loss = F.binary_cross_entropy_with_logits(
                    output_confs_fetched[one_point_mask],
                    input_masks_flatten[one_point_mask].float(),  # make the second point disappear (Not because there was no GT but for general regularization)
                )
            else:
                one_point_dist_loss = one_point_color_loss = one_point_conf_loss = 0 * torch.mean(output_points_occ)
            # If no GT point exist, just apply general regularization
            zero_point_mask = (input_points_len == 0)
            if torch.any(zero_point_mask):
                zero_point_conf_loss = F.binary_cross_entropy_with_logits(
                    output_confs_fetched[zero_point_mask],
                    input_masks_flatten[zero_point_mask].float(),  # make the two points disappear (Not because there was no GT but for general regularization)
                )
            else:
                zero_point_conf_loss = 0 * torch.mean(output_points_occ)

            total_loss = {
                "dist_loss": two_points_dist_loss + one_point_dist_loss,
                "color_loss": two_points_color_loss + one_point_color_loss,
                "conf_loss": two_points_conf_loss + one_point_conf_loss + zero_point_conf_loss,
            }

        else:
            raise NotImplementedError


        return total_loss
