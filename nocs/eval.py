import argparse
import glob
import os
from tqdm import tqdm
import pickle
import numpy as np
from utils.util import compute_degree_cm_mAP
from inference import synset_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', default='data/nocs_prediction', help='Directory for pose predictions')
    args = parser.parse_args()
    
    result_pkl_list = glob.glob(os.path.join(args.pred_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)[:]
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

    iou_3d_aps, pose_aps, pose_pred_matches, pose_gt_matches = compute_degree_cm_mAP(final_results, synset_names, args.pred_dir + '_map',
                                                            degree_thresholds = [5, 10, 15],
                                                            shift_thresholds= [5, 10, 15],
                                                            iou_3d_thresholds=np.linspace(0, 1, 101),
                                                            iou_pose_thres=0.1,
                                                            use_matches_for_pose=True)