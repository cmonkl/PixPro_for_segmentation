import torch
import torch.nn as nn
from typing import Tuple

# batch linspace
@torch.jit.script
def linspace(start: torch.Tensor, stop: torch.Tensor, num: int) -> torch.Tensor:
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    for i in range(start.ndim):
        steps = steps.unsqueeze(0)
    out = start[:, None] + steps * (stop - start)[:, None]
    return out


class ConsistencyLoss(nn.Module):
    """
    Loss function for original PixPro method
    """

    def __init__(self, distance_threshold: float):
        super(ConsistencyLoss, self).__init__()
        self.thresh = distance_threshold

    """
    Forward Pass
    
    Parameters:
        feat_q_view1, feat_q_view2: feat maps of both views from student network
        feat_k_view1, feat_k_view2: feat maps of both views from teacher network
       coords_view1, coords_view2: coordinates of patches 
    """

    def forward(
        self,
        feat_q_view1: torch.Tensor,
        feat_q_view2: torch.Tensor,
        feat_k_view1: torch.Tensor,
        feat_k_view2: torch.Tensor,
        coords_view1: torch.Tensor,
        coords_view2: torch.Tensor,
    ):
        N, C, H, W = feat_q_view1.shape

        # masks of positive pixel pairs from both views
        pos_pairs_mask_1 = self.compute_dist_matrix(H, W, coords_view1, coords_view2)
        pos_pairs_mask_2 = self.compute_dist_matrix(H, W, coords_view2, coords_view1)

        # don't compute loss if there's no intersection
        if (pos_pairs_mask_1 == 0).all() and (pos_pairs_mask_2 == 0).all():
            return None

        feat_q_view1 = feat_q_view1.view(N, C, -1)
        feat_k_view2 = feat_k_view2.view(N, C, -1)
        feat_q_view2 = feat_q_view2.view(N, C, -1)
        feat_k_view1 = feat_k_view1.view(N, C, -1)

        # similarity scores
        similarity_1 = torch.bmm(feat_q_view1.transpose(1, 2), feat_k_view2)
        similarity_2 = torch.bmm(feat_q_view2.transpose(1, 2), feat_k_view1)

        loss_1 = (similarity_1 * pos_pairs_mask_1).sum(-1).sum(-1) / (
            pos_pairs_mask_1.sum(-1).sum(-1) + 1e-6
        )
        loss_2 = (similarity_2 * pos_pairs_mask_2).sum(-1).sum(-1) / (
            pos_pairs_mask_2.sum(-1).sum(-1) + 1e-6
        )
        loss = -loss_1.mean() - loss_2.mean()

        # save data for logging
        log_dict = {"matches1": pos_pairs_mask_1, "matches2": pos_pairs_mask_2}
        return loss, log_dict

    def get_proj_coords(
        self, coord: torch.Tensor, H: int, W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_h, num_w = H, W
        offset_x = (coord[:, 3] - coord[:, 1]) / (2 * num_w)
        centers_x = linspace(coord[:, 1] + offset_x, coord[:, 3] - offset_x, num_w).to(
            coord.device
        )
        offset_y = (coord[:, 2] - coord[:, 0]) / (2 * num_h)
        centers_y = linspace(coord[:, 0] + offset_y, coord[:, 2] - offset_y, num_h).to(
            coord.device
        )
        grid_x = centers_x.unsqueeze(-2).expand(-1, num_h, -1)
        grid_y = centers_y.unsqueeze(-1).expand(-1, -1, num_w)
        proj_coords = torch.cat((grid_y.unsqueeze(-1), grid_x.unsqueeze(-1)), dim=3)
        bin_diag = torch.sqrt((offset_x * 2).pow(2) + (offset_y * 2).pow(2))

        return proj_coords, bin_diag

    def compute_dist_matrix(
        self,
        feat_map_H: int,
        feat_map_W: int,
        coords_1: torch.Tensor,
        coords_2: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = coords_1.size()[0]

        # get grid coordinates
        proj_coords_1, bin_diag_1 = self.get_proj_coords(
            coords_1, feat_map_H, feat_map_W
        )
        proj_coords_2, bin_diag_2 = self.get_proj_coords(
            coords_2, feat_map_H, feat_map_W
        )

        # calculate normalized distances
        distances = torch.cdist(
            proj_coords_1.view(batch_size, -1, 2), proj_coords_2.view(batch_size, -1, 2)
        )
        dist_matrix = distances / torch.max(bin_diag_1, bin_diag_2).unsqueeze(
            -1
        ).unsqueeze(-1)

        pos_dist_mask = (dist_matrix < self.thresh).float().detach()
        return pos_dist_mask


class PosConsistencyLoss(nn.Module):
    """
    Loss for positive pairs only
    """

    def __init__(self, distance_threshold: float):
        super(PosConsistencyLoss, self).__init__()
        self.thresh = distance_threshold

    def forward(
        self,
        feat_q_view1: torch.Tensor,
        feat_q_view2: torch.Tensor,
        feat_k_view1: torch.Tensor,
        feat_k_view2: torch.Tensor,
    ):
        sim = nn.CosineSimilarity()
        loss1 = sim(feat_q_view1, feat_k_view2)
        loss2 = sim(feat_q_view2, feat_k_view1)
        loss = (-loss1 - loss2).mean()

        return loss

