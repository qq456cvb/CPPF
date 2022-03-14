import glob
import open3d as o3d
from models.model import PPFEncoder, PointEncoder
import os
import pickle
import argparse
from torchvision.models import segmentation
from utils.util import backproject, fibonacci_sphere, convert_layers, estimate_normals

import cv2
import numpy as np
import omegaconf
import MinkowskiEngine as ME
import torch
from tqdm import tqdm
import cupy as cp
from models.voting import rot_voting_kernel, backvote_kernel, ppf_kernel

synset_names = ['BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ]

synset_names_inv = dict([(k, v) for v, k in enumerate(synset_names)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_dir', default='data/nocs_seg', help='Segmentation PKL files for NOCS')
    parser.add_argument('--nocs_dir', default='data/nocs', help='NOCS real test image path')
    parser.add_argument('--out_dir', default='data/nocs_prediction', help='Output directory for predictions')
    parser.add_argument('--cp_device', type=int, default=0, help='GPU device number for custom voting algorithms')
    parser.add_argument('--ckpt_path', default='checkpoints', help='Model checkpoint path')
    parser.add_argument('--angle_prec', type=float, default=1.5, help='Angle precision in orientation voting')
    parser.add_argument('--num_rots', type=int, default=72, help='Number of candidate center votes generated for a given point pair')
    parser.add_argument('--n_threads', type=int, default=512, help='Number of cupy threads')
    parser.add_argument('--bbox_mask', action='store_true', help='Whether to use bbox mask instead of instance segmentations')
    parser.add_argument('--adaptive_voting', action='store_true', help='Whether to use adaptive center voting')
    args = parser.parse_args()

    cp_device = args.cp_device
    result_pkl_list = glob.glob(os.path.join(args.seg_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)[:]
    
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    assert len(result_pkl_list)

    final_results = []
    for pkl_path in tqdm(result_pkl_list):
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
            
            if not 'gt_handle_visibility' in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                print('can\'t find gt_handle_visibility in the pkl.')
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(result['gt_handle_visibility'], result['gt_class_ids'])


        if type(result) is list:
            final_results += result
        elif type(result) is dict:
            final_results.append(result)
        else:
            assert False
    
    
    point_encoders = {}
    ppf_encoders = {}
    cfgs = {}
    for cls_id in range(1, 7):
        cls_name = synset_names[cls_id]
        path = os.path.join(args.ckpt_path, cls_name)
        nepoch = 'best'
        cfg = omegaconf.OmegaConf.load(f'{path}/.hydra/config.yaml')
        point_encoder = PointEncoder(k=cfg.knn, spfcs=[32, 64, 32, 32], num_layers=1, out_dim=32).cuda().eval()
        ppf_encoder = PPFEncoder(ppffcs=[84, 32, 32, 16], out_dim=2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins + 2 + 3).cuda().eval()
        
        cfgs[cls_name] = cfg
        
        point_encoder.load_state_dict(torch.load(f'{path}/point_encoder_epoch{nepoch}.pth'))
        ppf_encoder.load_state_dict(torch.load(f'{path}/ppf_encoder_epoch{nepoch}.pth'))
        
        point_encoders[cls_name] = point_encoder
        ppf_encoders[cls_name] = ppf_encoder
        
        if cls_name == 'laptop':
            laptop_aux = segmentation.fcn_resnet50(num_classes=2).cuda().eval()
            laptop_aux = convert_layers(laptop_aux, torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d).cuda().eval()
            laptop_aux.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'laptop_aux/segmenter_current.pth')))

    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    
    angle_tol = args.angle_prec
    num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
    sphere_pts = np.array(fibonacci_sphere(num_samples))

    num_rots = args.num_rots
    n_threads = args.n_threads
    bcelogits = torch.nn.BCEWithLogitsLoss()

    for res in tqdm(final_results):
        img = cv2.imread(os.path.join(args.nocs_dir, res['image_path'][5:] + '_color.png'))[:, :, ::-1]
        depth = cv2.imread(os.path.join(args.nocs_dir, res['image_path'][5:] + '_depth.png'), -1)

        bboxs = res['pred_bboxes']
        masks = res['pred_masks'].copy()
        RTs = np.zeros((len(bboxs), 4, 4), dtype=np.float32)
        for i in range(len(RTs)):
            RTs[i] = np.eye(4)
        scales = np.ones((len(bboxs), 3), dtype=np.float32)
        cls_ids = res['pred_class_ids']
        
        for i, bbox in enumerate(bboxs):
            if args.bbox_mask:
                masks[:, :, i][bbox[0]:bbox[2], bbox[1]:bbox[3]] = True

            cls_id = cls_ids[i]
            cls_name = synset_names[cls_id]
            
            cfg = cfgs[cls_name]
            point_encoder = point_encoders[cls_name]
            ppf_encoder = ppf_encoders[cls_name]
            
            pc, idxs = backproject(depth, intrinsics, masks[:, :, i])
            pc /= 1000
            # augment
            pc = pc + np.clip(cfg.res / 4 * np.random.randn(*pc.shape), -cfg.res / 2, cfg.res / 2)
            
            pc[:, 0] = -pc[:, 0]
            pc[:, 1] = -pc[:, 1]
            # pc = pc.astype(np.float32)
            
            high_res_indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=cfg.res)[1]
            pc = pc[high_res_indices].astype(np.float32)
            pc_normal = estimate_normals(pc, cfg.knn).astype(np.float32)
            
            if cls_name == 'laptop':
                mask_idxs = np.where(masks[:, :, i])
                bbox = np.array([
                    [np.min(mask_idxs[0]), np.max(mask_idxs[0])],
                    [np.min(mask_idxs[1]), np.max(mask_idxs[1])]
                ])
                
                rgb_obj = np.zeros_like(img, dtype=np.float32)
                rgb_obj[mask_idxs[0], mask_idxs[1]] = img[mask_idxs[0], mask_idxs[1]] / 255.
                rgb_cropped = cv2.resize(rgb_obj[bbox[0][0]:bbox[0][1]+1, bbox[1][0]:bbox[1][1]+1], (224, 224))
                resize_scale = 224 / (bbox[:, 1] - bbox[:, 0])
                
                pc_xy = np.stack(idxs, -1)
                idxs_resized = np.clip(((pc_xy - bbox[:, 0]) * resize_scale).astype(np.int64), 0, 223)
                
                output = laptop_aux(torch.from_numpy(rgb_cropped[None]).cuda().permute(0, 3, 1, 2))['out']
                preds_laptop_aux = output[0].argmax(0).cpu().numpy()
                pc_img_indices = idxs_resized[high_res_indices]
                # [high_res_indices]
                preds_laptop_aux = preds_laptop_aux[pc_img_indices[:, 0], pc_img_indices[:, 1]]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[preds_laptop_aux == 0])
                if (preds_laptop_aux == 0).sum() < 10:
                    laptop_up = None
                else:
                    plane, inlier = pcd.segment_plane(distance_threshold=0.02,
                                                ransac_n=3,
                                                num_iterations=100)
                    laptop_up = plane[:3]
            
            pcs = torch.from_numpy(pc[None]).cuda()
            pc_normals = torch.from_numpy(pc_normal[None]).cuda()
            
            point_idxs = np.random.randint(0, pc.shape[0], (100000, 2))
            
            with torch.no_grad():
                dist = torch.cdist(pcs, pcs)
                sprin_feat = point_encoder(pcs, pc_normals, dist)
                preds = ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs)
                preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
            
            preds_tr = torch.softmax(preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins), -1)
            preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
            preds_tr[0, :, 0] = preds_tr[0, :, 0] / (cfg.tr_num_bins - 1) * 2 * cfg.vote_range[0] - cfg.vote_range[0]
            preds_tr[0, :, 1] = preds_tr[0, :, 1] / (cfg.tr_num_bins - 1) * cfg.vote_range[1]
            
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
            RTs[i][:3, -1] = T_est
            
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
                    # sprin_feat = point_encoder.forward_nbrs(pc[None], pc_normal[None], torch.from_numpy(knn_idxs).cuda()[None])[0]
                    preds = ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs)
                    
                    preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
                    preds_up = preds[..., 2 * cfg.tr_num_bins:2 * cfg.tr_num_bins + cfg.rot_num_bins]
                    preds_right = preds[..., 2 * cfg.tr_num_bins + cfg.rot_num_bins:2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins]
                    preds_up_aux = preds[..., -5]
                    preds_right_aux = preds[..., -4]
                    preds_scale = preds[..., -3:]
                    
                    preds_tr = torch.softmax(preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins), -1)
                    preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
                    preds_tr[0, :, 0] = preds_tr[0, :, 0] / (cfg.tr_num_bins - 1) * 2 * cfg.vote_range[0] - cfg.vote_range[0]
                    preds_tr[0, :, 1] = preds_tr[0, :, 1] / (cfg.tr_num_bins - 1) * cfg.vote_range[1]
                    
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
                
                # vote for orientation
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
                    sph_cp = torch.tensor(sphere_pts.T, dtype=torch.float32).cuda()
                    start = np.arange(0, point_idxs.shape[0] * num_rots, num_rots)
                    np.random.shuffle(start)
                    sub_sample_idx = (start[:10000, None] + np.arange(num_rots)[None]).reshape(-1)
                    candidates = torch.as_tensor(candidates, device='cuda').reshape(-1, 3)
                    candidates = candidates[torch.LongTensor(sub_sample_idx).cuda()]
                    cos = candidates.mm(sph_cp)
                    counts = torch.sum(cos > np.cos(angle_tol / 180 * np.pi), 0).cpu().numpy()
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
            
            if (cls_name == 'laptop') and (laptop_up is not None):
                if np.dot(up, laptop_up) + np.dot(right, laptop_up) < np.dot(up, -laptop_up) + np.dot(right, -laptop_up):
                    laptop_up = -laptop_up
                
                # detect wrong orientation
                if np.dot(up, laptop_up) < np.dot(right, laptop_up):
                    right = up
                    up = laptop_up
                    right -= np.dot(up, right) * up
                    right /= (np.linalg.norm(right) + 1e-9)
            
            if np.linalg.norm(right) < 1e-7: # right is zero
                right = np.random.randn(3)
                right -= right.dot(up) * up
                right /= np.linalg.norm(right)

            if cfg.z_right:
                R_est = np.stack([np.cross(up, right), up, right], -1)
            else:
                R_est = np.stack([right, up, np.cross(right, up)], -1)

            pred_scale = np.exp(preds_scale[0].mean(0).cpu().numpy()) * cfg.scale_mean * 2
            scale_norm = np.linalg.norm(pred_scale)
            assert scale_norm > 0
            RTs[i][:3, :3] = R_est * scale_norm
            scales[i, :] = pred_scale / scale_norm
            
        res['pred_RTs'] = RTs
        res['pred_scales'] = scales
        
        out_path = os.path.join(out_dir, 'results_' + '_'.join(res['image_path'].split('/')[1:]) + '.pkl')
        pickle.dump(res, open(out_path, 'wb'))
