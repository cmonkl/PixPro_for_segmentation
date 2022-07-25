import torch
import torch.nn as nn
from pos_pairs import get_pos_pairs
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from resnet import MomentumBatchNorm2d
from unet import UNet
from typing import Dict


class Projection(nn.Module):
    """
        Projection Head for Encoder Network 
    """
    def __init__(self, in_feat: int, mid_feat: int, out_feat: int, bn_type="vanilla"):
        super(Projection, self).__init__()
        self.MLP = LinearBlock(in_feat, mid_feat, bn_type)
        self.conv2 = nn.Conv2d(mid_feat, out_feat, kernel_size=1)

    def forward(self, x):
        x = self.MLP(x)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, bn_type: str):
        super(Encoder, self).__init__()
        self.net = UNet(bn_type=bn_type)

    def forward(self, x, use_down_feat):
        x = self.net(x, use_down_feat)
        return x
        

class LocPosPPM4(nn.Module):
    """
    Pixel Propagation Module for only positive pairs and Local Region 
    """
    def __init__(self, params: Dict):
        super(LocPosPPM4, self).__init__()
        self.gamma = params.gamma
        self.transform = Transform(params.num_up_features, params.transform_layers)
        self.local_size = params.local_size

    def forward(self, x: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        # pos_idx: NxNumPos
        pos_lim = pos_idx.shape[1]
        transform_x = self.transform(x)
        transform_x = F.normalize(transform_x, dim=1) 
        x = F.normalize(x, dim=1)
        
        # pos pixels of x
        res1 = x[
            torch.repeat_interleave(torch.arange(N), pos_idx.size(1)),
            :,
            pos_idx.view(-1, 2)[:, 0],
            pos_idx.view(-1, 2)[:, 1],
        ].view(N, -1, C) # N npos C
        
        # indices of local regions
        bin_ind_h = torch.div(pos_idx[:, :, 0], self.local_size, rounding_mode="trunc")
        bin_ind_w = torch.div(pos_idx[:, :, 1], self.local_size, rounding_mode="trunc")

        # start indices
        start_ind_y = bin_ind_h * self.local_size
        start_ind_x = bin_ind_w * self.local_size
        arange = (
            torch.arange(self.local_size)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(N, pos_lim, 1)
            .to(start_ind_x.device)
        )
        # grid indices
        ind_x = arange + start_ind_x.unsqueeze(2)
        idx_x = torch.repeat_interleave(ind_x, self.local_size, dim=2)

        ind_y = arange + start_ind_y.unsqueeze(2)
        idx_y = ind_y.repeat(1, 1, self.local_size)

        # x local grid features
        x1 = x[
            torch.repeat_interleave(torch.arange(N), 
                                    pos_lim*self.local_size*self.local_size),
            :,
            idx_y.flatten(),
            idx_x.flatten(),
        ].view(N, pos_lim, self.local_size*self.local_size, C)
     
        # transformed x local grid features
        transform_x1 = transform_x[
            torch.repeat_interleave(torch.arange(N), 
                                    pos_lim*self.local_size*self.local_size),
            :,
            idx_y.flatten(),
            idx_x.flatten(),
        ].view(N, -1, C).permute(0, 2, 1) # N C nposlocloc
        
        sim = torch.matmul(res1.unsqueeze(2), x1.permute(0, 1, 3, 2)).squeeze()
        
        sim = torch.pow(torch.clamp(sim, min=0.), self.gamma)
        transform_x1 = transform_x1.view(N, 
                                         C, 
                                         pos_lim, 
                                         self.local_size*self.local_size)

        feat_q = torch.matmul(transform_x1.permute(0, 2, 1, 3), 
                              sim.unsqueeze(-1)).squeeze().permute(0, 2, 1)
        
        return feat_q

class LinearBlock(nn.Module):
    """
       Block of linear layer with ReLU
    """
    def __init__(self, in_feat: int, out_feat: int, bn_type="vanilla"):
        super(LinearBlock, self).__init__()
        self.conv = nn.Conv2d(in_feat, out_feat, kernel_size=1)
        if bn_type == "vanilla":
            self.bn = nn.BatchNorm2d(out_feat)
        else:
            self.bn = MomentumBatchNorm2d(out_feat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class Transform(nn.Module):
    """
        Transformation layers in Pixel Propagation Module
    """
    def __init__(self, num_feat: int, num_layers: int):
        super(Transform, self).__init__()
        self.blocks = nn.ModuleList(
            [LinearBlock(num_feat, num_feat) for i in range(num_layers-1)]
        )
        self.blocks.append(nn.Conv2d(num_feat, num_feat, kernel_size=1))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PPM(nn.Module):
    """
        Pixel Propagation Module consisting of similarity computation and linear transformation
    """
    def __init__(self, params: Dict):
        super(PPM, self).__init__()
        self.gamma = params.gamma
        self.transform = Transform(params.num_features, params.transform_layers)

    def forward(self, x, log_sim=False):
        N, C, H, W = x.shape
        # get transformed feats
        transform_x = self.transform(x)
        transform_x = F.normalize(transform_x, dim=1).view(N, C, -1)

        x = F.normalize(x, dim=1)
        x = x.view(N, C, -1)
	
	# compute similarities
        sim_matrix = torch.pow(torch.clamp(torch.bmm(x.transpose(1, 2), x), min=0.), self.gamma)
	
	# compute similarities * transform
        res = torch.bmm(transform_x, sim_matrix.transpose(1, 2)).view(N, C, H, W)

        if log_sim:
            return res, sim_matrix
        else:
            return res

class PixProLocPos(nn.Module):
    """
        Pixel Propagation Model with modification
        - computing only for positive pairs
        - use only local regions for propagation
        
        Parameters:
            params: setup config dict
            num_feat: dim of embedding vectors
            use_down_feat: whether to use original method PixPro on downsampled feats
            num_feat_down: dim of down feat embeddings
    """
    def __init__(self, params: Dict, num_feat=64, use_down_feat=True, num_feat_down=128):
        super(PixProLocPos, self).__init__()
        model_params = params.model
        self.use_down_feat = use_down_feat
        self.q_encoder = Encoder(params.student_model)
        self.q_PPM = LocPosPPM4(model_params.ppm_params)
        self.q_proj = Projection(*model_params.proj_layers_channels, bn_type="vanilla")
        if self.use_down_feat:
            self.q_down_proj = Projection(
                *model_params.down_proj_layers_channels, bn_type="vanilla"
            )
        self.PPM = PPM(model_params.ppm_params)

        self.k_encoder = Encoder(params.teacher_model)
        self.k_proj = Projection(*model_params.proj_layers_channels, bn_type="vanilla")
        if self.use_down_feat:
            self.k_down_proj = Projection(
                *model_params.down_proj_layers_channels, bn_type="vanilla"
            )
        self.momentum = model_params.momentum
	
	# total number of iterations
        self.K = (
            params.total_num_epochs
            * params.len_dataset
            / params.experiment_details.train_batch_size
        )
        
        # current iteration number
        self.k = (
            params.experiment_details.start_epoch
            * params.len_dataset
            / params.experiment_details.train_batch_size
        )

	# freeze gradient computation for teacher network
        for param_k, param_q in zip(
            self.k_encoder.parameters(), self.q_encoder.parameters()
        ):
            param_k.data = param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        for param_k, param_q in zip(self.k_proj.parameters(), self.q_proj.parameters()):
            param_k.data = param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        if self.use_down_feat:    
            for param_k, param_q in zip(self.k_down_proj.parameters(), self.q_down_proj.parameters()):
                param_k.data = param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
	
    # momentum update of teacher's network weights
    @torch.no_grad()
    def momentum_update(self):
        momentum = (
            1.0 - (1.0 - self.momentum) * (np.cos(np.pi * self.k / self.K) + 1.) / 2.0
        )
        self.k = self.k + 1

        for param_k, param_q in zip(
            self.k_encoder.parameters(), self.q_encoder.parameters()
        ):
            param_k.data = param_k.data * momentum + (1. - momentum) * param_q.data
        for param_k, param_q in zip(self.k_proj.parameters(), self.q_proj.parameters()):
            param_k.data = param_k.data * momentum + (1. - momentum) * param_q.data
        if self.use_down_feat:
            for param_k, param_q in zip(self.k_down_proj.parameters(), self.q_down_proj.parameters()):
                param_k.data = param_k.data * momentum + (1. - momentum) * param_q.data

    def forward(self, query: torch.Tensor, key: torch.Tensor, coord_q: torch.Tensor, coord_k: torch.Tensor, log_coord=False, log_down=False):
        N, C, H, W = query.shape
        
        # get positive pairs indexes for both views
        pos_pairs_idx1, pos_pairs_idx2, log_data_pos = get_pos_pairs(
            coord_q,
            coord_k,
            H,
            W,
            intersect_limit=60,
            num_pos=150,
            dist_thresh=0.7,
            proj=True,
            log_coord=log_coord,
        )  
        
        # mask for valid images (num pos >= thresh)
        batch_mask = log_data_pos["batch_mask"]
        intersect_num = log_data_pos["intersect_num"]
	
        batch_idx = (~batch_mask).nonzero().flatten()
        # use only valid images
        query = query[batch_idx].to(coord_q.device)
        key = key[batch_idx].to(coord_q.device)
	
	# get student network's feats
        if self.use_down_feat:
            q1, down_q1 = self.q_encoder(query, use_down_feat=True)
            proj_down_q1 = self.q_down_proj(down_q1)
            if log_down:
                proj_down_q1, sim1 = self.PPM(proj_down_q1, log_sim=log_down)
            else:
                proj_down_q1 = self.PPM(proj_down_q1, log_sim=log_down)
            proj_down_q1 = F.normalize(proj_down_q1, dim=1)
        else:
            q1 = self.q_encoder(query, use_down_feat=False)
        
        q1 = self.q_proj(q1)
        q1 = F.normalize(self.q_PPM(q1, pos_pairs_idx1), dim=1)

        if self.use_down_feat:
            q2, down_q2 = self.q_encoder(key, use_down_feat=True)
            proj_down_q2 = self.q_down_proj(down_q2)
            if log_down:
                proj_down_q2, sim2 = self.PPM(proj_down_q2, log_sim=log_down)
            else:
                proj_down_q2 = self.PPM(proj_down_q2, log_sim=log_down)
            proj_down_q2 = F.normalize(proj_down_q2, dim=1)
        else:
            q2 = self.q_encoder(key, use_down_feat=False)

        q2 = self.q_proj(q2)
        q2 = F.normalize(self.q_PPM(q2, pos_pairs_idx2), dim=1)
	
	# get teacher network's feats
        with torch.no_grad():
            self.momentum_update()
            if self.use_down_feat:
                k1, down_k1 = self.k_encoder(query, use_down_feat=True)
                proj_down_k1 = self.k_down_proj(down_k1)
                proj_down_k1 = F.normalize(proj_down_k1, dim=1)
            else:
                k1 = self.k_encoder(query, use_down_feat=False)
            k1 = self.k_proj(k1)
            k1 = F.normalize(k1, dim=1)

            if self.use_down_feat:
                k2, down_k2 = self.k_encoder(key, use_down_feat=True)
                proj_down_k2 = self.k_down_proj(down_k2)
                proj_down_k2 = F.normalize(proj_down_k2, dim=1)
            else:
                k2 = self.k_encoder(key, use_down_feat=False)
            k2 = self.k_proj(k2)
            k2 = F.normalize(k2, dim=1)

            N, C, H, W = k1.shape

	    # take only features of positive indexes
            k1 = k1[
                torch.repeat_interleave(torch.arange(pos_pairs_idx1.size(0)), 
                                        pos_pairs_idx1.size(1)),
                :,
                pos_pairs_idx1.view(-1, 2)[:, 0],
                pos_pairs_idx1.view(-1, 2)[:, 1],
            ].view(pos_pairs_idx1.size(0), -1, C).permute(0, 2, 1)

            k2 = k2[
                torch.repeat_interleave(torch.arange(pos_pairs_idx2.size(0)), 
                                        pos_pairs_idx2.size(1)),
                :,
                pos_pairs_idx2.view(-1, 2)[:, 0],
                pos_pairs_idx2.view(-1, 2)[:, 1],
            ].view(pos_pairs_idx2.size(0), -1, C).permute(0, 2, 1)
        
        res = q1, q2, k1, k2
        if self.use_down_feat:
            down_res = (proj_down_q1, proj_down_q2, proj_down_k1, proj_down_k2)
            new_coords = (coord_q[batch_idx], coord_k[batch_idx])
	
	# save data for logging
        log_data = {}
        log_data["N"] = N

        if log_coord:
            log_data["proj_coords1"] = log_data_pos["proj_coords1"].detach().cpu()
            log_data["proj_coords2"] = log_data_pos["proj_coords2"].detach().cpu()
            log_data["pos_coords1"] = log_data_pos["proj_pos_coords1"].detach().cpu()
            log_data["pos_coords2"] = log_data_pos["proj_pos_coords2"].detach().cpu()

        if self.use_down_feat and log_down:
            log_data["sim1"] = sim1.detach().cpu()
            log_data["sim2"] = sim2.detach().cpu()
        
        if self.use_down_feat:
            return res, down_res, new_coords, log_data
        else:
            return q1, q2, k1, k2, N
