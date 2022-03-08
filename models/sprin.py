import torch
import torch.nn as nn


def smooth_l1_loss(input, target, beta, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


class AlignToX(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    ## x: [..., 3], assume x normalied
    def forward(self, x):
        xn = x
        y2 = xn[..., 1] ** 2
        z2 = xn[..., 2] ** 2
        yz = xn[..., 1] * xn[..., 2]
        y2z2 = y2 + z2

        trans = torch.stack([
            xn[..., 0], xn[..., 1], xn[..., 2],
            -xn[..., 1], 1 + (xn[..., 0] - 1) * y2 / y2z2, (xn[..., 0] - 1) * yz / y2z2,
            -xn[..., 2], (xn[..., 0] - 1) * yz / y2z2, 1 + (xn[..., 0] - 1) * z2 / y2z2
        ], -1)
        
        return trans.reshape(*trans.shape[:-1], 3, 3)



def rifeat(points_r, points_s):
    """generate rotation invariant features
    Args:
        points_r (B x N x K x 3): 
        points_s (B x N x 1 x 3): 
    """

    # [*, 3] -> [*, 8] with compatible intra-shapes
    if points_r.shape[1] != points_s.shape[1]:
        points_r = points_r.expand(-1, points_s.shape[1], -1, -1)
    
    r_mean = torch.mean(points_r, -2, keepdim=True)
    l1, l2, l3 = r_mean - points_r, points_r - points_s, points_s - r_mean
    l1_norm = torch.norm(l1, 'fro', -1, True)
    l2_norm = torch.norm(l2, 'fro', -1, True)
    l3_norm = torch.norm(l3, 'fro', -1, True).expand_as(l2_norm)
    theta1 = (l1 * l2).sum(-1, keepdim=True) / (l1_norm * l2_norm + 1e-7)
    theta2 = (l2 * l3).sum(-1, keepdim=True) / (l2_norm * l3_norm + 1e-7)
    theta3 = (l3 * l1).sum(-1, keepdim=True) / (l3_norm * l1_norm + 1e-7)
    
    return torch.cat([l1_norm, l2_norm, l3_norm, theta1, theta2, theta3], dim=-1)


def conv_kernel(iunit, ounit, *hunits):
    layers = []
    for unit in hunits:
        layers.append(nn.Linear(iunit, unit))
        layers.append(nn.LayerNorm(unit))
        layers.append(nn.ReLU())
        iunit = unit
    layers.append(nn.Linear(iunit, ounit))
    return nn.Sequential(*layers)


class GlobalInfoProp(nn.Module):
    def __init__(self, n_in, n_global):
        super().__init__()
        self.linear = nn.Linear(n_in, n_global)

    def forward(self, feat):
        # [b, k, n_in] -> [b, k, n_in + n_global]
        tran = self.linear(feat)
        glob = tran.max(-2, keepdim=True)[0].expand(*feat.shape[:-1], tran.shape[-1])
        return torch.cat([feat, glob], -1)

        
class SparseSO3Conv(nn.Module):
    def __init__(self, rank, n_in, n_out, *kernel_interns, layer_norm=True):
        super().__init__()
        self.kernel = conv_kernel(6, rank, *kernel_interns)
        self.outnet = nn.Linear(rank * n_in, n_out)
        self.rank = rank
        self.layer_norm = nn.LayerNorm(n_out) if layer_norm else None

    def do_conv_ranked(self, r_inv_s, feat):
        # [b, n, k, rank], [b, n, k, cin] -> [b, n, cout]
        kern = self.kernel(r_inv_s).reshape(*feat.shape[:-1], self.rank)
        # PointConv-like optimization
        contracted = torch.einsum("bnkr,bnki->bnri", kern, feat).flatten(-2)
        return self.outnet(contracted)

    def forward(self, feat_points, feat, eval_points):
        eval_points_e = torch.unsqueeze(eval_points, -2)
        r_inv_s = rifeat(feat_points, eval_points_e)
        conv = self.do_conv_ranked(r_inv_s, feat)
        if self.layer_norm is not None:
            return self.layer_norm(conv)
        return conv