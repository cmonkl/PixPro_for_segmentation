import torch
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Dict, List
from matplotlib.pyplot import cm

"""
   Get denormalized version of an image
   with statistics from ImageNet
"""
def denormalize(image: torch.Tensor) -> torch.Tensor:
    res = (
        image * torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    ) + torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    return res


"""
   Convert coordinates from relative to absolute
"""
def relative_coords_to_abs(coord: torch.Tensor, im_h: int, im_w: int) -> torch.Tensor:
    coords = torch.zeros_like(coord).long()
    coords[:, 0] = torch.round(coord[:, 0] * (im_h - 1)).long()
    coords[:, 1] = torch.round(coord[:, 1] * (im_w - 1)).long()
    return coords


"""
   Draw coordinate boxes on original image
"""
def draw_coords(im: torch.Tensor, coords: torch.Tensor, color) -> torch.Tensor:
    for i, coord in enumerate(coords):
        if type(color) == type(torch.tensor([2, 3])):
            im[:, coord[0], coord[1]] = color
        else:
            im[:, coord[0], coord[1]] = torch.tensor(color[i][:3])
    return im


"""
   Check how much memory is allocated
"""
def get_gpu_memory_consumption(
    model: torch.Module, inputs: Tuple
) -> Tuple[float, float]:
    model.cpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    model_mem = model_memory * 2 / 1024 / 1024 / 1024

    sample_input1, sample_input2, coord1, coord2 = inputs

    output = model(
        sample_input1.to(device),
        sample_input2.to(device),
        coord1.to(device),
        coord2.to(device),
    )
    c = torch.cuda.memory_allocated(device)
    forward_pass_memory = c - b
    forward_pass_mem = forward_pass_memory / 1024 / 1024 / 1024
    print("model memory: ", model_mem)
    print("forward pass memory: ", forward_pass_mem)
    return model_mem, forward_pass_mem


"""
   Update model state dict
"""
def get_dict(d: Dict, cur_name="") -> Dict:
    res_dict = {}
    for k in d.keys():
        cur_key_name = cur_name + "_" if cur_name != "" else ""
        if type(d[k]) == type(d):
            res_dict.update(get_dict(d[k], cur_key_name + k))
        elif type(d[k]) == type([]):
            res_dict[cur_key_name + k] = "_".join(map(str, d[k]))
        else:
            res_dict[cur_key_name + k] = d[k]

    return res_dict


def draw_coord(img: torch.Tensor, old_coord: torch.Tensor):
    new_img = (transforms.ToTensor()(img)).clone()
    coord = old_coord.clone()
    if coord[3] < coord[1]:
        old_v1 = coord[1].item()
        old_v2 = coord[3].item()
        coord[1], coord[3] = old_v2, old_v1

    new_img[:, coord[0] : coord[2], coord[1] - 1 : coord[1] + 1] = (
        torch.Tensor([1, 0, 0]).unsqueeze(-1).unsqueeze(-1)
    )
    new_img[:, coord[0] : coord[2], coord[3] - 1 : coord[3] + 1] = (
        torch.Tensor([1, 0, 0]).unsqueeze(-1).unsqueeze(-1)
    )
    new_img[:, coord[0] - 1 : coord[0] + 1, coord[1] : coord[3]] = (
        torch.Tensor([1, 0, 0]).unsqueeze(-1).unsqueeze(-1)
    )
    new_img[:, coord[2] - 1 : coord[2] + 1, coord[1] : coord[3]] = (
        torch.Tensor([1, 0, 0]).unsqueeze(-1).unsqueeze(-1)
    )

    return new_img


"""
   Get images with coordinates drawn on them for logging
   
   Parameters:
       imgs: original images
       proj_coords1, proj_coords2: coordinates of downsampled patches
       pos_coords1, pos_coords2: positive pair coordinates from patches
  
   Return:
       ims: images with coordinate boxes of projective coords drawn
       ims_pos: images with positive pairs coordinates drawn
"""
def get_log_coord_imgs(
    imgs: torch.Tensor,
    proj_coords1: torch.Tensor,
    proj_coords2: torch.Tensor,
    pos_coords1: torch.Tensor,
    pos_coords2: torch.Tensor,
    log_limit=10,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    to_pil = transforms.ToPILImage()
    ims = []
    ims_pos = []
    for i, im in enumerate(imgs[:log_limit]):
        im_pos = im.clone()
        im_h, im_w = im.shape[1:]
        proj_abs1 = relative_coords_to_abs(proj_coords1[i].view(-1, 2), im_h, im_w)
        proj_abs2 = relative_coords_to_abs(proj_coords2[i].view(-1, 2), im_h, im_w)

        im = draw_coords(im, proj_abs1, torch.tensor([1.0, 0.0, 0.0]))
        im = draw_coords(im, proj_abs2, torch.tensor([0.0, 0.0, 1.0]))
        pos_coord1 = relative_coords_to_abs(pos_coords1[i], im_h, im_w)
        pos_coord2 = relative_coords_to_abs(pos_coords2[i], im_h, im_w)

        im_pos = draw_coords(im_pos, pos_coord1, torch.tensor([1.0, 0.0, 0.0]))
        im_pos = draw_coords(im_pos, pos_coord2, torch.tensor([0.0, 0.0, 1.0]))
        if pos_coord1[(pos_coord1 == pos_coord2).all(dim=1)].shape[0] > 0:
            im_pos = draw_coords(
                im_pos,
                pos_coord1[(pos_coord1 == pos_coord2).all(dim=1)],
                torch.tensor([0.0, 1.0, 0.0]),
            )
        ims.append(to_pil(im))
        ims_pos.append(to_pil(im_pos))

    return ims, ims_pos

