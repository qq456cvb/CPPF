import torch
import torch.nn as nn
import torch.nn.functional as F
from .sprin import GlobalInfoProp, SparseSO3Conv
import numpy as np


class ResLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bn=False) -> None:
        super().__init__()
        assert(bn is False)
        self.fc1 = torch.nn.Linear(dim_in, dim_out)
        if bn:
            self.bn1 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn1 = lambda x: x
        self.fc2 = torch.nn.Linear(dim_out, dim_out)
        if bn:
            self.bn2 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn2 = lambda x: x
        if dim_in != dim_out:
            self.fc0 = torch.nn.Linear(dim_in, dim_out)
        else:
            self.fc0 = None
    
    def forward(self, x):
        x_res = x if self.fc0 is None else self.fc0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x + x_res

    
class PointEncoder(nn.Module):
    def __init__(self, k, spfcs, out_dim, num_layers=2, num_nbr_feats=2) -> None:
        super().__init__()
        self.k = k
        self.spconvs = nn.ModuleList()
        self.spconvs.append(SparseSO3Conv(32, num_nbr_feats, out_dim, *spfcs))
        self.aggrs = nn.ModuleList()
        self.aggrs.append(GlobalInfoProp(out_dim, out_dim // 4))
        for _ in range(num_layers - 1):
            self.spconvs.append(SparseSO3Conv(32, out_dim + out_dim // 4, out_dim, *spfcs))
            self.aggrs.append(GlobalInfoProp(out_dim, out_dim // 4))

    def forward(self, pc, pc_normal, dist):
        nbrs_idx = torch.topk(dist, self.k, largest=False, sorted=False)[1]  #[..., N, K]
        pc_nbrs = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc.shape[-1]))  #[..., N, K, 3]
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  #[..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)
        
        pc_normal_nbrs = torch.gather(pc_normal.unsqueeze(-3).expand(*pc_normal.shape[:-1], *pc_normal.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc_normal.shape[-1]))  #[..., N, K, 3]
        pc_normal_cos = torch.sum(pc_normal_nbrs * pc_normal.unsqueeze(-2), -1, keepdim=True)
        
        feat = self.aggrs[0](self.spconvs[0](pc_nbrs, torch.cat([pc_nbrs_norm, pc_normal_cos], -1), pc))
        for i in range(len(self.spconvs) - 1):
            spconv = self.spconvs[i + 1]
            aggr = self.aggrs[i + 1]
            feat_nbrs = torch.gather(feat.unsqueeze(-3).expand(*feat.shape[:-1], *feat.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, feat.shape[-1]))
            feat = aggr(spconv(pc_nbrs, feat_nbrs, pc))
        return feat
    
    def forward_nbrs(self, pc, pc_normal, nbrs_idx):
        pc_nbrs = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc.shape[-1]))  #[..., N, K, 3]
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  #[..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)
        
        pc_normal_nbrs = torch.gather(pc_normal.unsqueeze(-3).expand(*pc_normal.shape[:-1], *pc_normal.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc_normal.shape[-1]))  #[..., N, K, 3]
        pc_normal_cos = torch.sum(pc_normal_nbrs * pc_normal.unsqueeze(-2), -1, keepdim=True)
        
        feat = self.aggrs[0](self.spconvs[0](pc_nbrs, torch.cat([pc_nbrs_norm, pc_normal_cos], -1), pc))
        for i in range(len(self.spconvs) - 1):
            spconv = self.spconvs[i + 1]
            aggr = self.aggrs[i + 1]
            feat_nbrs = torch.gather(feat.unsqueeze(-3).expand(*feat.shape[:-1], *feat.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, feat.shape[-1]))
            feat = aggr(spconv(pc_nbrs, feat_nbrs, pc))
        return feat


class PPFEncoder(nn.Module):
    def __init__(self, ppffcs, out_dim) -> None:
        super().__init__()
        self.res_layers = nn.ModuleList()
        for i in range(len(ppffcs) - 1):
            dim_in, dim_out = ppffcs[i], ppffcs[i + 1]
            self.res_layers.append(ResLayer(dim_in, dim_out, bn=False))
        self.final = nn.Linear(ppffcs[-1], out_dim)

    def forward(self, pc, pc_normal, feat, dist=None, idxs=None):
        if idxs is not None:
            return self.forward_with_idx(pc[0], pc_normal[0], feat[0], idxs)[None]
        xx = pc.unsqueeze(-2) - pc.unsqueeze(-3)
        xx_normed = xx / (dist[..., None] + 1e-7)

        outputs = []
        for idx in torch.chunk(torch.arange(pc.shape[1]), 5):
            feat_chunk = feat[..., idx, :]
            target_shape = [*feat_chunk.shape[:-2], feat_chunk.shape[-2], feat.shape[-2], feat_chunk.shape[-1]]  # B x NC x N x F
            xx_normed_chunk = xx_normed[..., idx, :, :]
            ppf = torch.cat([
                torch.sum(pc_normal[..., idx, :].unsqueeze(-2) * xx_normed_chunk, -1, keepdim=True), 
                torch.sum(pc_normal.unsqueeze(-3) * xx_normed_chunk, -1, keepdim=True), 
                torch.sum(pc_normal[..., idx, :].unsqueeze(-2) * pc_normal.unsqueeze(-3), -1, keepdim=True), 
                dist[..., idx, :, None],
            ], -1)
            # ppf.zero_()
            final_feat = torch.cat([feat_chunk[..., None, :].expand(*target_shape), feat[..., None, :, :].expand(*target_shape), ppf], -1)
        
            output = final_feat
            for res_layer in self.res_layers:
                output = res_layer(output)
            outputs.append(output)
        
        output = torch.cat(outputs, dim=-3)
        return self.final(output)

    def forward_with_idx(self, pc, pc_normal, feat, idxs):
        a_idxs = idxs[:, 0]
        b_idxs = idxs[:, 1]
        xy = pc[a_idxs] - pc[b_idxs]
        xy_norm = torch.norm(xy, dim=-1)
        xy_normed = xy / (xy_norm[..., None] + 1e-7)
        pnormal_cos = pc_normal[a_idxs] * pc_normal[b_idxs]
        ppf = torch.cat([
            torch.sum(pc_normal[a_idxs] * xy_normed, -1, keepdim=True),
            torch.sum(pc_normal[b_idxs] * xy_normed, -1, keepdim=True),
            torch.sum(pnormal_cos, -1, keepdim=True),
            xy_norm[..., None],
        ], -1)
        # ppf.zero_()
        
        final_feat = torch.cat([feat[a_idxs], feat[b_idxs], ppf], -1)
        
        output = final_feat
        for res_layer in self.res_layers:
            output = res_layer(output)
        return self.final(output)
