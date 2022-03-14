import os
from utils.util import compute_degree_cm_mAP
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import pickle
from utils.box import Box
from utils.iou import IoU
import argparse
from sunrgbd.inference import type2class


def iou_3d(box1, box2):
    try:
        return IoU(box1, box2).iou()
    except Exception as e:
        print(e)
        return 0
    
    
def nms(boxes, scores, overlap_threshold):
    I = np.argsort(scores)
    pick = []
    while (I.size != 0):
        last = I.size
        i = I[-1]
        pick.append(i)
        suppress = [last-1]
        for pos in range(last-1):
            j = I[pos]
            o = iou_3d(boxes[i], boxes[j])
            if (o>overlap_threshold):
                suppress.append(pos)
        I = np.delete(I,suppress)
    return pick


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', default='data/sunrgbd_prediction', help='Directory for pose predictions')
    parser.add_argument('--sunrgbd_dir', default='data/OFFICIAL_SUNRGBD', help='SUN RGB-D official data')
    parser.add_argument('--sunrgbd_extra_dir', default='data/sunrgbd_extra', help='SUN RGB-D extra file directory')
    parser.add_argument('--full_rot', action='store_true', help='Whether to evaluate full 3D rotations (default: gravity direction)')
    args = parser.parse_args()
    
    split_set = 'val'
    for typename in type2class:
        class_id = type2class[typename]
        
        scan_names = open(os.path.join(args.sunrgbd_extra_dir, 'scan_names_list/{}_{}.txt'.format(class_id, split_set))).readlines()
        scan_names = [scan_name.rstrip('\n') for scan_name in scan_names]
        
        l, h, w = 2, 2, 2
        bbox_raw = np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]]).T
        
        meta_data = sio.loadmat(os.path.join(args.sunrgbd_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'))['SUNRGBDMeta'][0]

        root_path = os.path.join(args.sunrgbd_extra_dir, 'sunrgbd_pc_bbox_votes_50k_v1_val')
        res_path = os.path.join(args.pred_dir, 'results_{}.pkl'.format(typename))
        poses_pred = pickle.load(open(res_path, 'rb'))
        final_results = []
        with tqdm(scan_names[:]) as t:
            for scan_name in t:
                meta = meta_data[int(scan_name) - 1]
                
                K = meta['K']
                Rtilt = meta['Rtilt']

                rot = np.load(os.path.join(root_path, scan_name) + '_rot.npy')
                Rtilt = rot @ Rtilt
                
                map_scene = {
                    'gt_class_ids': [],
                    'gt_RTs': [],
                    'gt_up_syms': [],
                    'gt_scales': [],
                    'pred_class_ids': [],
                    'pred_RTs': [],
                    'pred_scales': [],
                    'pred_scores': [],
                    'pred_bboxes': []
                }
                bboxes_gt = np.load(os.path.join(root_path, scan_name)+'_bbox.npy') # K,8
                
                ex_mat = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                extra_mat = ex_mat @ Rtilt @ ex_mat
                for bbox in bboxes_gt:
                    category = int(bbox[7])
                    tx, ty, tz, ry, sx, sy, sz = bbox[0], bbox[2], bbox[1], -bbox[6], bbox[3], bbox[5], bbox[4]
                    rot_gt =  np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]])
                    trans_gt = np.array([tx, ty, tz])
                    scales_gt = np.array([sx, sy, sz])
                    
                    RT = np.eye(4)
                    RT[:3, :3] = np.linalg.inv(extra_mat) @ ex_mat @ rot @ ex_mat @ rot_gt
                    RT[:3, 3] = np.linalg.inv(extra_mat) @ ex_mat @ rot @ ex_mat @ trans_gt
                    
                    bbox_pts = (rot_gt @ np.diag(scales_gt) @ bbox_raw.T).T + trans_gt
                    if category == class_id:
                        map_scene['gt_class_ids'].append(1)
                        map_scene['gt_RTs'].append(RT)
                        map_scene['gt_scales'].append(scales_gt)
                        map_scene['gt_up_syms'].append(False if args.full_rot else True)
                        
                map_scene['gt_RTs'] = np.stack(map_scene['gt_RTs'])
                map_scene['gt_scales'] = np.stack(map_scene['gt_scales'])
                map_scene['gt_up_syms'] = np.stack(map_scene['gt_up_syms'])
                        
                
                pose_pred = poses_pred[scan_name]
                boxes = []
                scores = []
                scale_ests = []
                rot_ests = []
                trans_ests = []
                for pred in pose_pred:
                    score = pred[1]
                    scale_est = np.array(pred[2:5])
                    rot_est = np.array(pred[5:14]).reshape(3, 3)
                    trans_est = np.array(pred[14:17])
                    if np.all(np.isfinite(pred)):
                        boxes.append(Box.from_transformation(rot_est, trans_est, scale_est))
                        scores.append(score)
                        scale_ests.append(scale_est)
                        rot_ests.append(rot_est)
                        trans_ests.append(trans_est)
                
                pick = nms(boxes, np.array(scores), 0.3)
                for i in pick:
                    map_scene['pred_class_ids'].append(1)
                    RT = np.eye(4)
                    RT[:3, :3] = np.linalg.inv(extra_mat) @ rot_ests[i]
                    RT[:3, 3] = np.linalg.inv(extra_mat) @ trans_ests[i]

                    map_scene['pred_RTs'].append(RT)
                    map_scene['pred_scales'].append(scale_ests[i])
                    map_scene['pred_scores'].append(scores[i])
                    map_scene['pred_bboxes'].append(boxes[i])
                
                if len(pick) > 0:
                    map_scene['pred_class_ids'] = np.stack(map_scene['pred_class_ids'])
                    map_scene['pred_RTs'] = np.stack(map_scene['pred_RTs'])
                    map_scene['pred_scales'] = np.stack(map_scene['pred_scales'])
                    map_scene['pred_scores'] = np.stack(map_scene['pred_scores'])
                    map_scene['pred_bboxes'] = np.stack(map_scene['pred_bboxes'])
                
                for k in map_scene:
                    map_scene[k] = np.array(map_scene[k])
                
                final_results.append(map_scene)

        print(f'Typename mAP: {typename}')
        compute_degree_cm_mAP(final_results, ['BG', typename], args.pred_dir + '_map', 
                            iou_3d_thresholds=np.linspace(0, 1, 101),
                            degree_thresholds = range(5, 61, 5), 
                            shift_thresholds=range(5, 31, 5),
                            iou_pose_thres=0.1,
                            use_matches_for_pose=True
                            )