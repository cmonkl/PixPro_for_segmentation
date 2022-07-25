import torch
import albumentations as A
from typing import Tuple, Dict, List

# batch linspace
@torch.jit.script
def linspace(start: torch.Tensor, stop: torch.Tensor, num: int) -> torch.Tensor:
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    for i in range(start.ndim):
        steps = steps.unsqueeze(0)
    out = start[:, None] + steps * (stop - start)[:, None]
    return out


"""
   Get grid coordinates
   
   Parameters:
       coord: coordinates of patch
       H, W: size of image
       proj: whether to use center coordinates
   
   Return:
       proj_coords: grid coordinates
       bin_diag: length of bin diagonal for normalizing distances
"""
def get_proj_coords(
    coord: torch.Tensor, H: int, W: int, proj=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_h, num_w = H, W

    # bin width
    interval_x = (coord[:, 3] - coord[:, 1]) / (num_w)
    # bin height
    interval_y = (coord[:, 2] - coord[:, 0]) / (num_h)

    # in case of using center projection
    if proj:
        interval_x = (coord[:, 3] - coord[:, 1]) / (2 * num_w)
        interval_y = (coord[:, 2] - coord[:, 0]) / (2 * num_h)

        centers_x = linspace(
            coord[:, 1] + interval_x, coord[:, 3] - interval_x, num_w
        ).to(coord.device)
        centers_y = linspace(
            coord[:, 0] + interval_y, coord[:, 2] - interval_y, num_h
        ).to(coord.device)
    else:
        interval_x = (coord[:, 3] - coord[:, 1]) / (num_w)
        interval_y = (coord[:, 2] - coord[:, 0]) / (num_h)

        centers_x = linspace(coord[:, 1], coord[:, 3], num_w).to(coord.device)
        centers_y = linspace(coord[:, 0], coord[:, 2], num_h).to(coord.device)

    grid_x = centers_x.unsqueeze(-2).expand(-1, num_h, -1)
    grid_y = centers_y.unsqueeze(-1).expand(-1, -1, num_w)
    proj_coords = torch.cat((grid_y.unsqueeze(-1), grid_x.unsqueeze(-1)), dim=3)
    if proj:
        bin_diag = torch.sqrt((interval_x * 2).pow(2) + (interval_y * 2).pow(2))
    else:
        bin_diag = torch.sqrt((interval_x).pow(2) + (interval_y).pow(2))

    return proj_coords, bin_diag


"""
   Get intersection mask for coordinates:
   True if proj_coord lies in the intersection with coord
   
   Parameters:
       proj_coords1: grid coordinates of one patch
       coord2: coordinates of another patch
       bin_diag: length of diagonal for normalizing distances
       thresh: threshold for positive pair distance
   
   Return:
       mask: boolean tensor with the same shape as proj_coords1
"""
def get_intersection_mask(
    proj_coords1: torch.Tensor,
    coord2: torch.Tensor,
    bin_diag: torch.Tensor,
    thresh: float,
) -> torch.Tensor:
    # epsilon for distance between positive pairs
    # print(bin_diag)
    y_eps = bin_diag * thresh
    x_eps = bin_diag * thresh

    y_tl_limit = (coord2[:, 0] - y_eps).unsqueeze(-1).unsqueeze(-1)
    y_br_limit = (coord2[:, 2] + y_eps).unsqueeze(-1).unsqueeze(-1)
    x_tl_limit = (
        (torch.min(coord2[:, 1], coord2[:, 3]) - x_eps).unsqueeze(-1).unsqueeze(-1)
    )
    x_br_limit = (
        (torch.max(coord2[:, 1], coord2[:, 3]) + x_eps).unsqueeze(-1).unsqueeze(-1)
    )

    y_tl_mask = proj_coords1[:, :, :, 0] >= y_tl_limit
    y_br_mask = proj_coords1[:, :, :, 0] <= y_br_limit
    x_tl_mask = proj_coords1[:, :, :, 1] >= x_tl_limit
    x_br_mask = proj_coords1[:, :, :, 1] <= x_br_limit

    mask = (y_tl_mask & y_br_mask & x_tl_mask & x_br_mask).detach()
    return mask


"""
   Get coordinates of positive pairs from two views
   
   Paramters:
       coord1, coord2: coordinates of views
       H, W: size of the image
       intersect_limit: limit of coordinates of the first view that fall into intersection
       num_pos: count of positive pairs
       dist_thresh: threshold for distances between positive coordinates
       proj: whether to use centering while projecting
       log_coord: whether to save data for logging
   Return:
       pos_pairs_view1, pos_pairs_view2: positive coordinates of both views
       log_dict: dictionary with data for logging
"""
def get_pos_pairs(
    coord1: torch.Tensor,
    coord2: torch.Tensor,
    H: int,
    W: int,
    intersect_limit=50,
    num_pos=100,
    dist_thresh=0.7,
    proj=True,
    log_coord=False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    # coords1, coords2: batchx4
    N, _ = coord1.shape

    # project coords to original image
    proj_coords1, bin_diag1 = get_proj_coords(coord1, H, W, proj=proj)
    proj_coords2, bin_diag2 = get_proj_coords(coord2, H, W, proj=proj)

    # intersection mask for the first and the second view
    mask1 = get_intersection_mask(proj_coords1, coord2, bin_diag2, dist_thresh)
    mask2 = get_intersection_mask(proj_coords2, coord1, bin_diag1, dist_thresh)

    # count the number of rows in intersection
    intersection_count = mask1.any(-1).sum(-1)
    skip_mask = intersection_count <= intersect_limit

    pos_pairs_view1 = []
    pos_pairs_view2 = []

    intersect_num = len(skip_mask) - sum(skip_mask).item()

    for i in range(len(skip_mask)):
        if not skip_mask[i]:
            intersection_idx1 = mask1[i].nonzero()

            # sample intersection idx (intersection might take entire image)
            unif = torch.ones(intersection_idx1.shape[0])
            idx = unif.multinomial(intersect_limit, replacement=False)
            samples = intersection_idx1[idx]
            intersection_coords1 = proj_coords1[i, samples[:, 0], samples[:, 1]]

            intersection_idx2 = mask2[i].nonzero()
            intersection_coords2 = proj_coords2[
                i, intersection_idx2[:, 0], intersection_idx2[:, 1]
            ]

            # account for difference in scale
            bin_diag = torch.max(bin_diag1, bin_diag2).unsqueeze(-1)[i]
            # normalized distances
            dist_matrix = (
                torch.cdist(intersection_coords1, intersection_coords2) / bin_diag
            )
            # indexes of pos pairs (row = for the first view, column = for the second view)
            mask_idx_1, mask_idx_2 = (dist_matrix < dist_thresh).nonzero(as_tuple=True)

            if len(mask_idx_1) < num_pos:
                skip_mask[i] = True
                continue

            unif = torch.ones(mask_idx_1.shape[0])
            sample_idx = unif.multinomial(num_pos, replacement=False)
            pos_pairs_idx_1 = mask_idx_1[sample_idx]
            pos_pairs_idx_2 = mask_idx_2[sample_idx]

            # pos pairs idx from intersection idx
            pos_pairs_1 = samples[pos_pairs_idx_1]
            pos_pairs_2 = intersection_idx2[pos_pairs_idx_2]

            pos_pairs_view1.append(pos_pairs_1)
            pos_pairs_view2.append(pos_pairs_2)

    pos_pairs_view1 = torch.stack(pos_pairs_view1)
    pos_pairs_view2 = torch.stack(pos_pairs_view2)

    if log_coord:
        bs, num_pairs, _ = pos_pairs_view1.shape
        log_coords1 = proj_coords1[~skip_mask][
            torch.arange(bs).repeat_interleave(num_pairs),
            pos_pairs_view1[:, :, 0].flatten(),
            pos_pairs_view1[:, :, 1].flatten(),
        ].view(pos_pairs_view1.shape)

        log_coords2 = proj_coords2[~skip_mask][
            torch.arange(bs).repeat_interleave(num_pairs),
            pos_pairs_view2[:, :, 0].flatten(),
            pos_pairs_view2[:, :, 1].flatten(),
        ].view(pos_pairs_view2.shape)

        log_dict = {
            "batch_mask": skip_mask,
            "intersect_num": intersect_num,
            "proj_coords1": proj_coords1[~skip_mask],
            "proj_coords2": proj_coords2[~skip_mask],
            "proj_pos_coords1": log_coords1,
            "proj_pos_coords2": log_coords2,
        }
    else:
        log_dict = {"batch_mask": skip_mask, "intersect_num": intersect_num}
    return pos_pairs_view1, pos_pairs_view2, log_dict


"""
   Get positive pair coordinates for UNet architecture with padding
"""
def get_pos_pairs_pad(
    proj_coords1: torch.Tensor,
    proj_coords2: torch.Tensor,
    intersect_limit=30,
    num_pos=100,
    dist_thresh=0.7,
    proj=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    pad = 92
    coord1 = torch.stack(
        [
            proj_coords1[:, pad, pad, 0],
            proj_coords1[:, pad, pad, 1],
            proj_coords1[:, -pad - 1, -pad - 1, 0],
            proj_coords1[:, -pad - 1, -pad - 1, 1],
        ],
        dim=1,
    )

    coord2 = torch.stack(
        [
            proj_coords2[:, pad, pad, 0],
            proj_coords2[:, pad, pad, 1],
            proj_coords2[:, -pad - 1, -pad - 1, 0],
            proj_coords2[:, -pad - 1, -pad - 1, 1],
        ],
        dim=1,
    )

    interval_x_1 = torch.abs(
        (proj_coords1[:, 0, 1:, 1] - proj_coords1[:, 0, :-1, 1])
    ).mean(1)
    interval_y_1 = torch.abs(
        (proj_coords1[:, 1:, 1, 0] - proj_coords1[:, :-1, 0, 0])
    ).min(1)[0]
    bin_diag1 = torch.sqrt((interval_x_1).pow(2) + (interval_y_1).pow(2))

    interval_x_2 = torch.abs(
        (proj_coords2[:, 0, 1:, 1] - proj_coords2[:, 0, :-1, 1])
    ).mean(1)
    interval_y_2 = torch.abs(
        (proj_coords2[:, 1:, 1, 0] - proj_coords2[:, :-1, 0, 0])
    ).min(1)[0]
    bin_diag2 = torch.sqrt((interval_x_2).pow(2) + (interval_y_2).pow(2))

    bin_diag = torch.max(bin_diag1, bin_diag2)

    # intersection mask for the first and the second view
    mask1 = get_intersection_mask(proj_coords1, coord2, bin_diag, dist_thresh)
    mask2 = get_intersection_mask(proj_coords2, coord1, bin_diag, dist_thresh)

    intersection_count = mask1.any(-2).sum(-1)
    skip_mask = intersection_count <= intersect_limit

    pos_pairs_view1 = []
    pos_pairs_view2 = []

    intersect_num = len(skip_mask) - sum(skip_mask).item()

    pos_pairs_view1 = []
    pos_pairs_view2 = []

    intersect_num = len(skip_mask) - sum(skip_mask).item()
    for i in range(len(skip_mask)):
        if not skip_mask[i]:
            intersection_idx1 = mask1[i].nonzero()

            # sample intersection idx (intersection might take entire image)
            unif = torch.ones(intersection_idx1.shape[0])
            idx = unif.multinomial(intersect_limit, replacement=False)
            samples = intersection_idx1[idx]
            intersection_coords1 = proj_coords1[i, samples[:, 0], samples[:, 1]]

            intersection_idx2 = mask2[i].nonzero()
            intersection_coords2 = proj_coords2[
                i, intersection_idx2[:, 0], intersection_idx2[:, 1]
            ]

            # account for difference in scale
            bin_diag = torch.max(bin_diag1, bin_diag2).unsqueeze(-1)[i]
            # normalized distances
            dist_matrix = (
                torch.cdist(intersection_coords1, intersection_coords2) / bin_diag
            )
            # indexes of pos pairs (row = for the first view, column = for the second view)
            pos_pairs_idx_1, pos_pairs_idx_2 = (dist_matrix < dist_thresh).nonzero(
                as_tuple=True
            )

            if len(pos_pairs_idx_1) < num_pos:
                skip_mask[i] = True
                continue

            unif = torch.ones(pos_pairs_idx_1.shape[0])
            sample_idx = unif.multinomial(num_pos, replacement=False)
            pos_pairs_idx_1 = pos_pairs_idx_1[sample_idx]
            pos_pairs_idx_2 = pos_pairs_idx_2[sample_idx]

            # pos pairs idx from intersection idx
            pos_pairs_1 = samples[pos_pairs_idx_1]
            pos_pairs_2 = intersection_idx2[pos_pairs_idx_2]

            pos_pairs_view1.append(pos_pairs_1)
            pos_pairs_view2.append(pos_pairs_2)

            del dist_matrix

    pos_pairs_view1 = torch.stack(pos_pairs_view1)
    pos_pairs_view2 = torch.stack(pos_pairs_view2)
    return pos_pairs_view1, pos_pairs_view2, skip_mask, intersect_num

