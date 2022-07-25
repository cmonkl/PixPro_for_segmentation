import torch.nn.functional as F
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple, Any, List


class RetrievalDataset(torch.utils.data.Dataset):
    """
    ImageNet Dataset for pixel retrieval
    """
    def __init__(self, root_dir: str, img_dir: str, img_size: int):
        self.img_dir = os.path.join(root_dir, img_dir)
        self.img_paths = [path for path in os.listdir(self.img_dir)]
        
        self.transform = transforms.Compose([ 
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.query_paths = [os.path.join(self.img_dir, 'n01440764_5927.JPEG'), 
                os.path.join(self.img_dir, 'n01491361_5817.JPEG'),
                os.path.join(self.img_dir, 'n01514668_13179.JPEG'), 
                os.path.join(self.img_dir, 'n01530575_30464.JPEG'),
                os.path.join(self.img_dir, 'n01531178_17067.JPEG')]
        
        self.query_imgs = [self.transform(Image.open(path)) for path in self.query_paths]
        relative_coords = torch.tensor([[126, 58], [160, 79], [103, 61], [215, 39], [50, 158]]) / 228
        self.query_coords = torch.round(relative_coords * img_size).long()
            
    def __len__(self):
        return len(self.img_paths)
    
    def denormalize(self, image: torch.Tensor):
        res = (image * torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)) + \
                  torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
        return res

    def __getitem__(self, idx):
        img_meta = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_meta)

        orig_img = Image.open(img_path).convert('RGB')
        img = self.transform(orig_img)
            
        return img

# argmax for top three values
def argmax123(inp: torch.Tensor) -> Tuple[Tuple, Tuple, Tuple]:
    # transform index to desired coordinates
    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))
        
    N = inp.shape[0]
    new_inp = torch.clone(inp)
    
    # find first argmax and change its score to -inf
    first_max_ind = unravel_index(torch.argmax(new_inp.view(N, -1), dim=1), 
                                  new_inp.shape[1:])
    new_inp[torch.arange(N), first_max_ind[0], 
                             first_max_ind[1], 
                             first_max_ind[2]] = -float('inf')
    
    # find second argmax and change its score to -inf
    second_max_ind = unravel_index(torch.argmax(new_inp.view(N, -1), dim=1), 
                                   new_inp.shape[1:])
    new_inp[torch.arange(N), second_max_ind[0], 
                             second_max_ind[1], 
                             second_max_ind[2]] = -float('inf')
    
    # find third argmax
    third_max_ind = unravel_index(torch.argmax(new_inp.view(N, -1), dim=1), 
                                   new_inp.shape[1:])
    return first_max_ind, second_max_ind, third_max_ind


# get similarity scores and indexes of the closest pairs
def get_closest(sim: torch.Tensor, img: torch.Tensor, feats: torch.Tensor, batch_size: int, ind: troch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = sim[torch.arange(batch_size), ind[0], ind[1], ind[2]]
    closest_idx = torch.stack((ind[0], ind[1], ind[2])).T
  
    return scores, closest_idx

# draw coordinate box
def draw_frame(img: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    color = torch.Tensor([1., 0,  0])
    t_img = img.permute(1, 2, 0)
    t_img[coord[0]-1:coord[0]+1, coord[1]-1:coord[1]+1] = color

    # draw rectangle around pixel
    square_size = int(min(t_img.shape[0], t_img.shape[1]) * 0.1)
    t = max(coord[0]-square_size, 0)
    b = min(coord[0]+square_size, t_img.shape[0]-1)
    l = max(coord[1]-square_size, 0)
    r = min(coord[1]+square_size, t_img.shape[1]-1)
    t_img[t:b, l] = color
    t_img[t:b, r] = color
    t_img[t, l:r] = color
    t_img[b, l:r] = color
    return t_img.permute(2, 0, 1)

# draw coordinate box for downsampled feats
def draw_frame_down(img: torch.Tensor, inp_coord: torch.Tensor, bin_h: int, bin_w: int) -> torch.Tensor:
    coords = inp_coord.clone()
    color = torch.Tensor([1., 0,  0])
    t_img = img.permute(1, 2, 0)
    # draw rectangle around pixel
    
    t = max(torch.round(coords[0]*bin_h).long(), 0)
    b = min(torch.round((coords[0]+1)*bin_h).long(), t_img.shape[0]-1)
    l = max(torch.round(coords[1]*bin_w).long(), 0)
    r = min(torch.round((coords[1]+1)*bin_w).long(), t_img.shape[1]-1)

    t_img[t:b, l] = color
    t_img[t:b, r] = color
    t_img[t, l:r] = color
    t_img[b, l:r] = color
    return t_img.permute(2, 0, 1)

"""
   Find pixels with closest vector embedding to the query pixel
"""
def pixel_retrieval(model: torch.nn.Module, 
                    retrieval_dataset: torch.utils.data.Dataset, 
                    dataloader: torch.utils.data.DataLoader, 
                    device: torch.device, 
                    batch_size: int,
                    logger: Any,
                    it: int,
                    query_num=5, 
                    top_num=3, feat_num=64) -> List[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # change calculations to eval mode
    model.eval()
    img_size = retrieval_dataset[0].shape[-1]
    model = model.to(device)


    with torch.no_grad():
        # get query pixel embeddings
        query_pixels = torch.zeros((query_num, feat_num))
        for i in range(len(retrieval_dataset.query_imgs)):
            img = retrieval_dataset.query_imgs[i].unsqueeze(0).to(device)
            feats = model(img)
            coords = retrieval_dataset.query_coords[i]
            query_pixels[i] = feats[:, :, coords[0], coords[1]].flatten()
	
	# index of top results
        closest_idx = torch.zeros((query_num, top_num, 3))
        # scores of similarity
        scores = torch.zeros((query_num, top_num))
        # batch number for each result
        batch_num = torch.zeros((query_num, top_num))

        i = 0
        for img in tqdm(dataloader):
            img = img.to(device)
            # get embeddings
            feats = model(img)
            sim = F.cosine_similarity(feats.unsqueeze(0),
                                  query_pixels.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(device), dim=2)
            # find closest indexes
            ind1, ind2, ind3 = argmax123(sim)
            new_scores = scores.clone().detach()
            new_closest_idx = closest_idx.clone().detach()
            new_batch_num = batch_num.clone()

            res0 = get_closest(sim, img, feats, query_num, ind1) 
            new_scores[:, 0], new_closest_idx[:, 0] = res0

            res1 = get_closest(sim, img, feats, query_num, ind2)
            new_scores[:, 1], new_closest_idx[:, 1] = res1

            res2 = get_closest(sim, img, feats, query_num, ind3)
            new_scores[:, 2], new_closest_idx[:, 2] = res2

            new_batch_num[:, :] = i
	
	    # keep only maximum scores
            mask = new_scores > scores
            scores[mask] = new_scores[mask]
            closest_idx[mask] = new_closest_idx[mask]
            batch_num[mask] = new_batch_num[mask]
            i+=1
    
    # get images by their indexes
    imgs = torch.zeros((query_num, top_num, 3, img_size, img_size))
    sim_scores = torch.zeros((query_num, top_num, img_size, img_size))
    for i, img in enumerate(tqdm(dataloader)):
        if (batch_num == i).any():
            mask = batch_num == i
            idx = closest_idx[mask].long()
            imgs[mask] = img[idx[:, 0]].cpu()
            with torch.no_grad():
                model.eval()
                batch_im = img[idx[:, 0]]
                assert len(batch_im.shape) == 4
                if len(batch_im.shape) < 4:
                    batch_im = batch_im.unsqueeze(0)
                feats = model(batch_im.to(device))
                q_pixels = query_pixels[mask.nonzero(as_tuple=True)[0]]
                
                sim = F.cosine_similarity(feats, q_pixels.unsqueeze(-1).unsqueeze(-1).to(device))
                sim_scores[mask] = sim.cpu()
    
    # save data for logging
    to_pil = transforms.ToPILImage()
    denormalize = retrieval_dataset.denormalize
    for i in range(query_num):
        key_imgs = [draw_frame(denormalize(img), 
                               closest_idx[i][j][1:].long()) for j, img in enumerate(imgs[i])]
        sim = [to_pil(F.relu(img)) for img in sim_scores[i]]
        framed_q = draw_frame(denormalize(retrieval_dataset.query_imgs[i]),
                             retrieval_dataset.query_coords[i])
        logger.log({f'query_{i+1}':logger.Image(to_pil(framed_q))})
        logger.log({f"key_{i+1}":[logger.Image(to_pil(image), 
                    caption=f'score={round(scores[i][j].item(), 3)}') for j, image in enumerate(key_imgs)]})
        logger.log({f"sim_{i+1}":[logger.Image(image, 
                    caption=f'score={round(scores[i][j].item(), 3)}') for j, image in enumerate(sim)]})
    return imgs, sim_scores, scores

# change coordinates to bins
def project_coords(coords, bin_h, bin_w):
    ind_y = torch.div(coords[:, 0], bin_h, rounding_mode='trunc')
    ind_x = torch.div(coords[:, 1], bin_w, rounding_mode='trunc')
    return torch.stack([ind_y, ind_x], dim=1).long()
