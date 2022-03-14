import sys
from models.model import PPFEncoder, PointEncoder
import scipy.io as sio
import torch
import os
import pickle

from tqdm import tqdm
import cv2
import numpy as np
import omegaconf
from utils.util import fibonacci_sphere, estimate_normals
import cupy as cp
from models.voting import backvote_kernel, rot_voting_kernel, ppf_kernel
import MinkowskiEngine as ME
import argparse


def backproject_sunrgbd(depth, K, Rtilt, mask=None):
    if mask is None:
        mask = np.ones_like(depth, dtype=bool)
    y, x = np.where(mask)
    xy = np.stack([x, y], -1)
    z = depth[mask] / 1000.
    xy = (xy - np.array([K[0, 2], K[1, 2]])) * z[..., None] / np.array([K[0, 0], K[1, 1]])
    points3d = np.stack([xy[:, 0], z, -xy[:, 1]], -1)
    points3d = (Rtilt @ points3d.T).T
    points3d = points3d[z != 0]
    points3d = points3d[:, [0, 2, 1]]
    return points3d


type2class = {'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sunrgbd_dir', default='data/OFFICIAL_SUNRGBD', help='SUN RGB-D official data')
    parser.add_argument('--sunrgbd_extra_dir', default='data/sunrgbd_extra', help='SUN RGB-D extra file directory')
    parser.add_argument('--out_dir', default='data/sunrgbd_prediction', help='Output directory for predictions')
    parser.add_argument('--cp_device', type=int, default=0, help='GPU device number for custom voting algorithms')
    parser.add_argument('--ckpt_path', default='checkpoints', help='Model checkpoint path')
    parser.add_argument('--angle_prec', type=float, default=1.5, help='Angle precision in orientation voting')
    parser.add_argument('--num_rots', type=int, default=72, help='Number of candidate center votes generated for a given point pair')
    parser.add_argument('--n_threads', type=int, default=512, help='Number of cupy threads')
    parser.add_argument('--adaptive_voting', action='store_true', help='Whether to use adaptive center voting')
    
    args = parser.parse_args()
    cp_device = args.cp_device
    n_threads = args.n_threads
    num_rots = args.num_rots
    angle_tol = args.angle_prec
    split_set = 'val'
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    for typename in type2class:
        class_id = type2class[typename]
        
        scan_names = open(os.path.join(args.sunrgbd_extra_dir, 'scan_names_list/{}_{}.txt'.format(class_id, split_set))).readlines()
        scan_names = [scan_name.rstrip('\n') for scan_name in scan_names]
        
        l, h, w = 2, 2, 2
        bbox_raw = np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]]).T
        
        meta_data = sio.loadmat(os.path.join(args.sunrgbd_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'))['SUNRGBDMeta'][0]
        path = os.path.join(args.ckpt_path, typename)
        nepoch = 'best'
        
        cfg = omegaconf.OmegaConf.load(f'{path}/.hydra/config.yaml')
        res = cfg.res
        point_encoder = PointEncoder(k=cfg.knn, num_layers=1, spfcs=[32, 64, 32, 32], out_dim=32).cuda().eval()
        ppf_encoder = PPFEncoder(ppffcs=[84, 32, 32, 16], out_dim=2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins + 2 + 3).cuda().eval()
        
        point_encoder.load_state_dict(torch.load(f'{path}/point_encoder_epoch{nepoch}.pth'))
        ppf_encoder.load_state_dict(torch.load(f'{path}/ppf_encoder_epoch{nepoch}.pth'))
        
        num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
        sphere_pts = np.array(fibonacci_sphere(num_samples))
        bcelogits = torch.nn.BCEWithLogitsLoss()
        
        vote_range = cfg.vote_range
        scale_mean = cfg.scale_mean
        
        root_path = os.path.join(args.sunrgbd_extra_dir, 'sunrgbd_pc_bbox_votes_50k_v1_val')
        poses_pred = {}

        category_scans = set()
        with tqdm(scan_names[:]) as t:
            for scan_name in t:
                meta = meta_data[int(scan_name) - 1]
                
                K = meta['K']
                Rtilt = meta['Rtilt']
                
                # random rotation
                rot = np.load(os.path.join(root_path, scan_name) + '_rot.npy')
                Rtilt = rot @ Rtilt

                # load_depth
                depth_path = meta['depthpath'][0]
                depth_path = depth_path[17:]
                depth_path = os.path.join(args.sunrgbd_dir, depth_path)
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                depth = (depth >> 3) | (depth << 13)
                depth[depth > 8000] = 8000
                
                bboxes_gt = np.load(os.path.join(root_path, scan_name)+'_bbox.npy') # K,8
                segments_gt = pickle.load(open(os.path.join(root_path, scan_name) + '_segments.pkl', 'rb'))
                ex_mat = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                points3d = (ex_mat @ rot @ np.load(os.path.join(root_path, scan_name)+'_pc.npz')['pc'][:, :3].T).T # Nx6
                
                pose_pred = []
                for i, bbox in enumerate(bboxes_gt):
                    if int(bbox[7]) == class_id:
                        category_scans.add(scan_name)
                        pc = points3d[segments_gt[i]]
                        
                        if pc.shape[0] < 200: 
                            continue
                        
                        if pc.shape[0] > 40000:
                            pc = pc[:40000]
                            
                        # augment
                        pc = pc + np.clip(cfg.res / 4 * np.random.randn(*pc.shape), -cfg.res / 2, cfg.res / 2)
                        
                        high_res_indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=cfg.res)[1]
                        pc = pc[high_res_indices].astype(np.float32)
                        pc_normal = estimate_normals(pc, cfg.knn).astype(np.float32)
                        
                        pcs = torch.from_numpy(pc[None]).cuda()
                        pc_normals = torch.from_numpy(pc_normal[None]).cuda()
                        
                        point_idxs = np.random.randint(0, pc.shape[0], (100000, 2))
                        
                        with torch.no_grad():
                            dist = torch.cdist(pcs, pcs)
                            if dist.shape[-1] < point_encoder.k:
                                continue
                            sprin_feat = point_encoder(pcs, pc_normals, dist)
                            preds = ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs)
                            preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
                        
                        preds_tr = torch.softmax(preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins), -1)
                        preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
                        preds_tr[0, :, 0] = preds_tr[0, :, 0] / (cfg.tr_num_bins - 1) * 2 * vote_range[0] - vote_range[0]
                        preds_tr[0, :, 1] = preds_tr[0, :, 1] / (cfg.tr_num_bins - 1) * vote_range[1]
                        
                        # vote for center
                        with cp.cuda.Device(cp_device):
                            block_size = (pc.shape[0] ** 2 + 512 - 1) // 512

                            corners = np.stack([np.min(pc, 0), np.max(pc, 0)])
                            grid_res = ((corners[1] - corners[0]) / cfg.res).astype(np.int32) + 1
                            grid_obj = cp.asarray(np.zeros(grid_res, dtype=np.float32))
                            ppf_kernel(
                                (block_size, 1, 1),
                                (512, 1, 1),
                                (
                                    cp.asarray(pc).astype(cp.float32), cp.asarray(preds_tr[0].cpu().numpy()).astype(cp.float32), cp.asarray(np.ones((pc.shape[0],))).astype(cp.float32), 
                                    cp.asarray(point_idxs).astype(cp.int32), grid_obj, cp.asarray(corners[0]), cp.float32(cfg.res), 
                                    point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2], True if args.adaptive_voting else False
                                )
                            )
                            
                            grid_obj = grid_obj.get()
                            cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
                            cand_world = corners[0] + cand * cfg.res

                        T_est = cand_world[-1]
                        corners = np.stack([np.min(pc, 0), np.max(pc, 0)])
                        
                        # back vote filtering
                        block_size = (point_idxs.shape[0] + n_threads - 1) // n_threads
                
                        pred_center = T_est
                        with cp.cuda.Device(cp_device):
                            output_ocs = cp.zeros((point_idxs.shape[0], 3), cp.float32)
                            backvote_kernel(
                                (block_size, 1, 1),
                                (n_threads, 1, 1),
                                (
                                    cp.asarray(pc), cp.asarray(preds_tr[0].cpu().numpy()), output_ocs, cp.asarray(point_idxs).astype(cp.int32), cp.asarray(corners[0]), cp.float32(cfg.res), 
                                    point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2], cp.asarray(pred_center).astype(cp.float32), cp.float32(3 * cfg.res)
                                )
                            )
                        oc = output_ocs.get()
                        mask = np.any(oc != 0, -1)
                        point_idxs = point_idxs[mask]
                        
                        with torch.cuda.device(0):
                            with torch.no_grad():
                                preds = ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs)
                                
                                preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
                                preds_up = preds[..., 2 * cfg.tr_num_bins:2 * cfg.tr_num_bins + cfg.rot_num_bins]
                                preds_right = preds[..., 2 * cfg.tr_num_bins + cfg.rot_num_bins:2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins]
                                preds_up_aux = preds[..., -5]
                                preds_right_aux = preds[..., -4]
                                preds_scale = preds[..., -3:]
                                
                                preds_tr = torch.softmax(preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins), -1)
                                preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
                                preds_tr[0, :, 0] = preds_tr[0, :, 0] / (cfg.tr_num_bins - 1) * 2 * vote_range[0] - vote_range[0]
                                preds_tr[0, :, 1] = preds_tr[0, :, 1] / (cfg.tr_num_bins - 1) * vote_range[1]
                                
                                preds_up = torch.softmax(preds_up[0], -1)
                                preds_up = torch.multinomial(preds_up, 1).float()[None]
                                preds_up[0] = preds_up[0] / (cfg.rot_num_bins - 1) * np.pi
                                
                                preds_right = torch.softmax(preds_right[0], -1)
                                preds_right = torch.multinomial(preds_right, 1).float()[None]
                                preds_right[0] = preds_right[0] / (cfg.rot_num_bins - 1) * np.pi

                        final_directions = []
                        for j, (direction, aux) in enumerate(zip([preds_up, preds_right], [preds_up_aux, preds_right_aux])):
                            if j == 1 and not cfg.regress_right:
                                continue
                            with cp.cuda.Device(cp_device):
                                candidates = cp.zeros((point_idxs.shape[0], num_rots, 3), cp.float32)

                                block_size = (point_idxs.shape[0] + 512 - 1) // 512
                                rot_voting_kernel(
                                    (block_size, 1, 1),
                                    (512, 1, 1),
                                    (
                                        cp.asarray(pc), cp.asarray(preds_tr[0].cpu().numpy()), cp.asarray(direction[0].cpu().numpy()), candidates, cp.asarray(point_idxs).astype(cp.int32), cp.asarray(corners[0]).astype(cp.float32), cp.float32(cfg.res), 
                                        point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2]
                                    )
                                )
                            candidates = candidates.get().reshape(-1, 3)
                            start = np.arange(0, point_idxs.shape[0] * num_rots, num_rots)
                            np.random.shuffle(start)
                            sub_sample_idx = (start[:10000, None] + np.arange(num_rots)[None]).reshape(-1)
                            candidates = candidates[sub_sample_idx]
                            
                            cos = np.matmul(candidates, sphere_pts.T)
                            counts = np.sum(cos > np.cos(angle_tol / 180 * np.pi), 0)
                            best_dir = np.array(sphere_pts[np.argmax(counts)])
                            
                            # filter up
                            ab = pc[point_idxs[:, 0]] - pc[point_idxs[:, 1]]
                            distsq = np.sum(ab ** 2, -1)
                            ab_normed = ab / (np.sqrt(distsq) + 1e-7)[..., None]
                            
                            pairwise_normals = pc_normal[point_idxs[:, 0]]
                            pairwise_normals[np.sum(pairwise_normals * ab_normed, -1) < 0] *= -1
                            
                            with torch.no_grad():
                                target = torch.from_numpy((np.sum(pairwise_normals * best_dir, -1) > 0).astype(np.float32)).cuda()
                                up_loss = bcelogits(aux[0], target).item()
                                down_loss = bcelogits(aux[0], 1. - target).item()
                                
                            if down_loss < up_loss:
                                final_dir = -best_dir
                            else:
                                final_dir = best_dir
                            final_directions.append(final_dir)
                        
                        up = final_directions[0]
                        if cfg.regress_right:
                            right = final_directions[1]
                            right -= np.dot(up, right) * up
                            right /= (np.linalg.norm(right) + 1e-9)
                        else:
                            right = np.array([0, -up[2], up[1]])
                            right /= (np.linalg.norm(right) + 1e-9)
                        
                        if np.linalg.norm(right) < 1e-7: # right is zero
                            right = np.random.randn(3)
                            right -= right.dot(up) * up
                            right /= (np.linalg.norm(right) + 1e-9)
                                
                        if cfg.z_right:
                            R_est = np.stack([np.cross(up, right), up, right], -1)
                        else:
                            R_est = np.stack([right, up, np.cross(right, up)], -1)

                        scale_est = np.exp(preds_scale[0].mean(0).cpu().numpy()) * scale_mean
                        bbox_mat = np.eye(4)
                        bbox_mat[:3, :3] = R_est @ np.diag(scale_est)
                        bbox_mat[:3, 3] = T_est
                        bbox = (bbox_mat @ np.concatenate([bbox_raw, np.ones([bbox_raw.shape[0], 1])], -1).T).T[:, :3]
                        
                        pose_pred.append([class_id, 1., *scale_est, *R_est.reshape(-1), *T_est])
                        
                poses_pred[scan_name] = pose_pred
                
        with open(os.path.join(args.out_dir, 'results_{}.pkl'.format(typename)), 'wb') as f:
            pickle.dump(poses_pred, f)