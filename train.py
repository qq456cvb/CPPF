from glob import glob

import hydra
import torch
from models.model import PPFEncoder, PointEncoder
from utils.dataset import ShapeNetDataset
import numpy as np
import logging
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from utils.util import AverageMeter, typename2shapenetid
import os
        
    
@hydra.main(config_path='./config', config_name='config')
def main(cfg):
    logger = logging.getLogger(__name__)
    
    # load name
    name_path = hydra.utils.to_absolute_path('data/shapenet_names/{}.txt'.format(cfg.category))
    if os.path.exists(name_path):
        shapenames = open(name_path).read().splitlines()
    else:
        shapenet_id = typename2shapenetid[cfg.category]
        shapenames = os.listdir(os.path.join(hydra.utils.to_absolute_path(cfg.shapenet_root), '{}'.format(shapenet_id)))
        shapenames = [shapenet_id + '/' + name for name in shapenames]
        
    ds = ShapeNetDataset(cfg, shapenames)
    df = torch.utils.data.DataLoader(ds, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=10)
    assert cfg.batch_size == 1
    
    point_encoder = PointEncoder(k=cfg.knn, spfcs=[32, 64, 32, 32], num_layers=1, out_dim=32).cuda()
    ppf_encoder = PPFEncoder(ppffcs=[84, 32, 32, 16], out_dim=2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins + 2 + 3).cuda()
    
    opt = optim.Adam([*point_encoder.parameters(), *ppf_encoder.parameters()], lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
    kldiv = nn.KLDivLoss(reduction='batchmean')
    bcelogits = nn.BCEWithLogitsLoss()
    
    logger.info('Train')
    best_loss = np.inf
    for epoch in range(cfg.max_epoch):
        n = 0
        
        loss_meter = AverageMeter()
        loss_tr_meter = AverageMeter()
        loss_up_meter = AverageMeter()
        loss_up_aux_meter = AverageMeter()
        loss_right_meter = AverageMeter()
        loss_right_aux_meter = AverageMeter()
        loss_scale_meter = AverageMeter()
        point_encoder.train()
        ppf_encoder.train()
        with tqdm(df) as t:
            for pcs, pc_normals, targets_tr, targets_rot, targets_rot_aux, targets_scale, point_idxs in t:
                pcs, pc_normals, targets_tr, targets_rot, targets_rot_aux, targets_scale, point_idxs = \
                    pcs.cuda(), pc_normals.cuda(), targets_tr.cuda(), targets_rot.cuda(), targets_rot_aux.cuda(), targets_scale.cuda(), point_idxs.cuda()
                
                opt.zero_grad()
                with torch.no_grad():
                    dist = torch.cdist(pcs, pcs)
                
                sprin_feat = point_encoder(pcs, pc_normals, dist)
                
                preds = ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs[0])
                
                preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
                preds_up = preds[..., 2 * cfg.tr_num_bins:2 * cfg.tr_num_bins + cfg.rot_num_bins]
                preds_right = preds[..., 2 * cfg.tr_num_bins + cfg.rot_num_bins:2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins]
                
                preds_up_aux = preds[..., -5]
                preds_right_aux = preds[..., -4]
                
                preds_scale = preds[..., -3:]

                loss_tr = kldiv(F.log_softmax(preds_tr[:, 0], dim=-1), targets_tr[0, :, 0]) + kldiv(F.log_softmax(preds_tr[:, 1], dim=-1), targets_tr[0, :, 1])
                loss_up = kldiv(F.log_softmax(preds_up[0], dim=-1), targets_rot[0, :, 0])
                loss_up_aux = bcelogits(preds_up_aux[0], targets_rot_aux[0, :, 0])
                loss_scale = F.mse_loss(preds_scale, targets_scale[:, None])
                
                loss = loss_up + loss_tr + loss_up_aux + loss_scale
                if cfg.regress_right:
                    loss_right = kldiv(F.log_softmax(preds_right[0], dim=-1), targets_rot[0, :, 1])
                    loss_right_aux = bcelogits(preds_right_aux[0], targets_rot_aux[0, :, 1])
                    
                    loss += loss_right + loss_right_aux
                    loss_right_meter.update(loss_right.item())
                    loss_right_aux_meter.update(loss_right_aux.item())
                    
                loss.backward(retain_graph=False)
                
                # torch.nn.utils.clip_grad_norm_([*point_encoder.parameters(), *ppf_encoder.parameters()], 1.)
                opt.step()
                
                loss_meter.update(loss.item())
                loss_tr_meter.update(loss_tr.item())
                loss_up_meter.update(loss_up.item())
                loss_up_aux_meter.update(loss_up_aux.item())
                loss_scale_meter.update(loss_scale.item())
                    
                n += 1
                if cfg.regress_right:
                    t.set_postfix(loss=loss_meter.avg, loss_tr=loss_tr_meter.avg, 
                            loss_up=loss_up_meter.avg, loss_right=loss_right_meter.avg,
                            loss_up_aux=loss_up_aux_meter.avg, loss_right_aux=loss_right_aux_meter.avg,
                            loss_scale=loss_scale_meter.avg)
                else:
                    t.set_postfix(loss=loss_meter.avg, loss_tr=loss_tr_meter.avg, 
                            loss_up=loss_up_meter.avg,
                            oss_up_aux=loss_up_aux_meter.avg,
                            loss_scale=loss_scale_meter.avg)
        if epoch % 20 == 0:
            torch.save(point_encoder.state_dict(), f'point_encoder_epoch{epoch}.pth')
            torch.save(ppf_encoder.state_dict(), f'ppf_encoder_epoch{epoch}.pth')
            
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg 
            torch.save(point_encoder.state_dict(), f'point_encoder_epochbest.pth')
            torch.save(ppf_encoder.state_dict(), f'ppf_encoder_epochbest.pth')
        logger.info('loss: {:.4f}, loss_tr: {:.4f}, loss_up: {:.4f}, loss_right: {:.4f}, loss_up_aux: {:.4f}, loss_right_aux: {:.4f}, loss_scale: {:.4f}'
                    .format(loss_meter.avg, loss_tr_meter.avg, loss_up_meter.avg, loss_right_meter.avg, loss_up_aux_meter.avg, loss_right_aux_meter.avg, loss_scale_meter.avg))

if __name__ == '__main__':
    main()