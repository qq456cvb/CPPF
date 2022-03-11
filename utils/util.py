"""
Mask R-CNN
Common utility functions and classes.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import open3d as o3d
from copyreg import pickle
import sys
import os
import cv2
import math
import random
import numpy as np
import pickle
from tqdm import tqdm
from ctypes import *
import time

import torch

from utils.aligning import estimateSimilarityTransform

import matplotlib.pyplot as plt
import math
from .box import Box
from .iou import IoU


typename2shapenetid = {
    'chair': '03001627',
    'bathtub': '02808440',
    'bookshelf': '02871439',
    'bed': '02818832',
    'sofa': '04256520',
    'table': '04379243'
}


def convert_layers(model, layer_type_old, layer_type_new, convert_weights=False):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_layers(module, layer_type_old, layer_type_new, convert_weights)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = layer_type_new(module.num_features, module.eps, module.momentum, module.affine,
                                             module.track_running_stats) 

            if convert_weights == True:
                layer_new.weight = layer_old.weight
                layer_new.bias = layer_old.bias

            model._modules[name] = layer_new

    return model


def estimate_normals(pc, knn):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.array(pcd.normals)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def roty(a):
    return np.array([[np.cos(a), 0, -np.sin(a), 0],
                        [0, 1, 0, 0],
                        [np.sin(a), 0, np.cos(a), 0],
                        [0, 0, 0, 1]])
    
def rotx(a):
    return np.array([[1, 0, 0, 0],
                        [0, np.cos(a), -np.sin(a), 0],
                        [0, np.sin(a), np.cos(a), 0],
                        [0, 0, 0, 1]])



def fibonacci_sphere(samples):
    
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def real2prob(val, max_val, num_bins, circular=False):
    is_torch = isinstance(val, torch.Tensor)
    if is_torch:
        res = torch.zeros((*val.shape, num_bins), dtype=val.dtype).to(val.device)
    else:
        res = np.zeros((*val.shape, num_bins), dtype=val.dtype)
        
    if not circular:
        interval = max_val / (num_bins - 1)
        if is_torch:
            low = torch.clamp(torch.floor(val / interval).long(), max=num_bins - 2)
        else:
            low = np.clip(np.floor(val / interval).astype(np.int64), a_min=None, a_max=num_bins - 2)
        high = low + 1
        # assert torch.all(low >= 0) and torch.all(high < num_bins)
        
        # huge memory
        if is_torch:
            res.scatter_(-1, low[..., None], torch.unsqueeze(1. - (val / interval - low), -1))
            res.scatter_(-1, high[..., None], 1. - torch.gather(res, -1, low[..., None]))
        else:
            np.put_along_axis(res, low[..., None], np.expand_dims(1. - (val / interval - low), -1), -1)
            np.put_along_axis(res, high[..., None], 1. - np.take_along_axis(res, low[..., None], -1), -1)
        # res[..., low] = 1. - (val / interval - low)
        # res[..., high] = 1. - res[..., low]
        # assert torch.all(0 <= res[..., low]) and torch.all(1 >= res[..., low])
        return res
    else:
        interval = max_val / num_bins
        if is_torch:
            val_new = torch.clone(val)
        else:
            val_new = val.copy()
        val_new[val < interval / 2] += max_val
        res = real2prob(val_new - interval / 2, max_val, num_bins + 1)
        res[..., 0] += res[..., -1]
        return res[..., :-1]


def prob2real(prob, max_val, num_bins, circular=False):
    is_torch = isinstance(prob, torch.Tensor)
    if not circular:
        if is_torch:
            return torch.sum(prob * torch.arange(num_bins).to(prob) * max_val / (num_bins - 1), -1)
        else:
            return np.sum(prob * np.arange(num_bins) * max_val / (num_bins - 1), -1)
    else:
        interval = max_val / num_bins
        if is_torch:
            vecs = torch.stack([torch.cos(torch.arange(num_bins).to(prob) * interval + interval / 2), torch.sin(torch.arange(num_bins).to(prob) * interval + interval / 2)], -1)
            res = torch.sum(prob[..., None] * vecs, -2)
            res = torch.atan2(res[..., 1], res[..., 0])
        else:
            vecs = np.stack([np.cos(np.arange(num_bins) * interval + interval / 2), np.sin(np.arange(num_bins) * interval + interval / 2)], -1)
            res = np.sum(prob[..., None] * vecs, -2)
            res = np.arctan2(res[..., 1], res[..., 0])
        res[res < 0] += 2 * np.pi # remap to [0, 2pi]
        return res


def compute_3d_iou(RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2):
    '''Computes IoU overlaps between two 3d bboxes.
       bbox_3d_1, bbox_3d_1: [3, 8]
    '''
    
    def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
        try:
            # import pdb; pdb.set_trace()
            RT_1[:3, :3] = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
            RT_2[:3, :3] = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
            box1 = Box.from_transformation(RT_1[:3, :3], RT_1[:3, -1], scales_1)
            box2 = Box.from_transformation(RT_2[:3, :3], RT_2[:3, -1], scales_2)
            return IoU(box1, box2).iou()
        except:
            return 0


    if RT_1 is None or RT_2 is None:
        return -1

    symmetry_flag = False
    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility==0):
        def y_rotation_matrix(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                             [0, 1, 0 , 0], 
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0, 0, 0 , 1]])

        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = RT_1 @ y_rotation_matrix(2*math.pi*i/float(n))
            max_iou = max(max_iou, asymmetric_3d_iou(rotated_RT_1, RT_2, scales_1, scales_2))
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)
    
    
    return max_iou


def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, handle_visibility, synset_names):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]
    
    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    '''

    ## make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    if synset_names[class_id] in ['bottle', 'can', 'bowl']:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['mug', 'chair', 'bathtub', 'bookshelf', 'bed', 'sofa', 'table'] and handle_visibility == 0:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result


def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]
    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d



def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

 


############################################################
#  Evaluation
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x

def compute_3d_matches(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                       pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
                       iou_3d_thresholds, score_threshold=0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)
    
    if num_pred:
        if len(pred_boxes.shape) == 2:
            pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        
        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            #overlaps[i, j] = compute_3d_iou(pred_3d_bboxs[i], gt_3d_bboxs[j], gt_handle_visibility[j], 
            #    synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])
            overlaps[i, j] = compute_3d_iou(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j], gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])

    # Loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(len(pred_boxes)):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                #print('gt_match: ', gt_match[j])
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                #print('iou: ', iou)
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue

                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break

    return gt_matches, pred_matches, overlaps, indices


def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)
    assert pred_match.shape[0] == pred_scores.shape[0]

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match  = pred_match[score_indices]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


def compute_RT_overlaps(gt_class_ids, gt_RTs, gt_handle_visibility,
                        pred_class_ids, pred_RTs, 
                        synset_names):
    """Finds overlaps between prediction and ground truth instances.
    Returns:
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # print('num of gt instances: {}, num of pred instances: {}'.format(len(gt_class_ids), len(gt_class_ids)))
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_degree_cm_symmetry(pred_RTs[i], 
                                                              gt_RTs[j], 
                                                              gt_class_ids[j], 
                                                              gt_handle_visibility[j],
                                                              synset_names)
            
    return overlaps


def compute_match_from_degree_cm(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)


    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches


    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2
    

    for d, degree_thres in enumerate(degree_thres_list):                
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Remove low scores
                # low_score_idx = np.where(sum_degree_shift >= 100)[0]
                # if low_score_idx.size > 0:
                #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
                # 3. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    #print(j, len(gt_match), len(pred_class_ids), len(gt_class_ids))
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue

                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches









############################################################
#  Miscellaneous
############################################################

def draw(img, imgpts, axes, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)


    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)


    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)

    
    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)


    # draw axes
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)  # z
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)  # x
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3) ## y last


    return img


def draw_text(draw_image, bbox, text, draw_box=False):
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    thickness = 1
    

    retval, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    
    bbox_margin = 10
    text_margin = 10
    
    text_box_pos_tl = (min(bbox[1] + bbox_margin, 635 - retval[0] - 2* text_margin) , min(bbox[2] + bbox_margin, 475 - retval[1] - 2* text_margin)) 
    text_box_pos_br = (text_box_pos_tl[0] + retval[0] + 2* text_margin,  text_box_pos_tl[1] + retval[1] + 2* text_margin)

    # text_pose is the bottom-left corner of the text
    text_pos = (text_box_pos_tl[0] + text_margin, text_box_pos_br[1] - text_margin - 3)
    
    if draw_box:
        cv2.rectangle(draw_image, 
                      (bbox[1], bbox[0]),
                      (bbox[3], bbox[2]),
                      (255, 0, 0), 2)

    cv2.rectangle(draw_image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (255,0,0), -1)
    
    cv2.rectangle(draw_image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (0,0,0), 1)

    cv2.putText(draw_image, text, text_pos,
                fontFace, fontScale, (255,255,255), thickness)

    return draw_image
    

def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    x = np.arange(width)
    y = np.arange(height)

    #non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    
    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz) #[num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]

    return pts, idxs


def align(class_ids, masks, coords, depth, intrinsics, synset_names, image_path, save_path=None, if_norm=False, with_scale=True, verbose=False):
    num_instances = len(class_ids)
    error_messages = ''
    elapses = []
    if num_instances == 0:
        return np.zeros((0, 4, 4)), np.ones((0, 3)), error_messages, elapses

    RTs = np.zeros((num_instances, 4, 4))
    bbox_scales = np.ones((num_instances, 3))
    
    for i in range(num_instances):
        class_name = synset_names[class_ids[i]]
        class_id = class_ids[i]
        mask = masks[:, :, i]
        coord = coords[:, :, i, :]
        abs_coord_pts = np.abs(coord[mask==1] - 0.5)
        bbox_scales[i, :] = 2*np.amax(abs_coord_pts, axis=0)

        pts, idxs = backproject(depth, intrinsics, mask)
        coord_pts = coord[idxs[0], idxs[1], :] - 0.5

        if if_norm:
            scale = np.linalg.norm(bbox_scales[i, :])
            bbox_scales[i, :] /= scale
            coord_pts /= scale

        try:
            start = time.time()
            
            scales, rotation, translation, outtransform = estimateSimilarityTransform(coord_pts, pts, False)

            aligned_RT = np.zeros((4, 4), dtype=np.float32) 
            if with_scale:
                aligned_RT[:3, :3] = np.diag(scales) / 1000 @ rotation.transpose()
            else:
                aligned_RT[:3, :3] = rotation.transpose()
            aligned_RT[:3, 3] = translation / 1000
            aligned_RT[3, 3] = 1
            
            if save_path is not None:
                coord_pts_rotated = aligned_RT[:3, :3] @ coord_pts.transpose() + aligned_RT[:3, 3:]
                coord_pts_rotated = coord_pts_rotated.transpose()
                np.savetxt(save_path+'_{}_{}_depth_pts.txt'.format(i, class_name), pts)
                np.savetxt(save_path+'_{}_{}_coord_pts.txt'.format(i, class_name), coord_pts)
                np.savetxt(save_path+'_{}_{}_coord_pts_aligned.txt'.format(i, class_name), coord_pts_rotated)

            if verbose:
                print('Mask ID: ', i)
                print('Scale: ', scales/1000)
                print('Rotation: ', rotation.transpose())
                print('Translation: ', translation/1000)

            elapsed = time.time() - start
            # print('elapsed: ', elapsed)
            elapses.append(elapsed)
        

        except Exception as e:
            message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(synset_names[class_id], image_path, str(e))
            print(message)
            error_messages += message + '\n'
            aligned_RT = np.identity(4, dtype=np.float32) 

        # print('Estimation takes {:03f}s.'.format(time.time() - start))
        # from camera world to computer vision frame
        z_180_RT = np.zeros((4, 4), dtype=np.float32)
        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
        z_180_RT[3, 3] = 1

        RTs[i, :, :] = z_180_RT @ aligned_RT 

    return RTs, bbox_scales, error_messages, elapses



def compute_degree_cm_mAP(final_results, synset_names, log_dir, degree_thresholds=[360], shift_thresholds=[100], iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_pred_scores_all  = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_gt_matches_all   = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    
    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
    pose_gt_matches_all  = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
    pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]

    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    
    pose_gt_matches = np.full((num_degree_thres, num_shift_thres, len(final_results), 20), -1, dtype=int)
    pose_pred_matches = np.full((num_degree_thres, num_shift_thres, len(final_results), 20), -1, dtype=int)
    for progress, result in tqdm(enumerate(final_results), total=len(final_results)):
        # print(progress, len(final_results))
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        # normalize RTs and scales
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']
        norm_gt_scales = np.stack([np.cbrt(np.linalg.det(gt_RT[:3, :3])) for gt_RT in gt_RTs])
        gt_RTs[:, :3, :3] = gt_RTs[:, :3, :3] / norm_gt_scales[:, None, None]
        gt_scales = gt_scales * norm_gt_scales[:, None]
    
        pred_bboxes = np.array(result['pred_bboxes'])
        pred_class_ids = result['pred_class_ids']
        pred_scales = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_RTs = np.array(result['pred_RTs'])
        pred_bboxes[...] = 1
        if len(pred_RTs) > 0:
            norm_pred_scales = np.stack([np.cbrt(np.linalg.det(pred_RT[:3, :3])) for pred_RT in pred_RTs])
            pred_RTs[:, :3, :3] = pred_RTs[:, :3, :3] / (norm_pred_scales[:, None, None] + 1e-9)
            pred_scales = pred_scales * norm_pred_scales[:, None]
        #print(pred_bboxes.shape[0], pred_class_ids.shape[0], pred_scores.shape[0], pred_RTs.shape[0])

        # import pdb; pdb.set_trace()
        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue


        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            if len(gt_class_ids) > 0:
                gt_idx_mapping = dict([(i, j) for i, j in enumerate(np.where(gt_class_ids==cls_id)[0])])
            else:
                gt_idx_mapping = dict([(i, j) for i, j in enumerate(range(20))])
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))

            if len(pred_class_ids) > 0:
                pred_idx_mapping = dict([(i, j) for i, j in enumerate(np.where(pred_class_ids==cls_id)[0])])
            else:
                pred_idx_mapping = dict([(i, j) for i, j in enumerate(range(20))])
            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_bboxes =  pred_bboxes[pred_class_ids==cls_id, :] if len(pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))

            # calculate the overlap between each gt instance and pred instance
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)


            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                           cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                           iou_thres_list)
            if len(iou_pred_indices):
                pred_idx_mapping = dict([(i, pred_idx_mapping[j]) for i, j in enumerate(iou_pred_indices)])
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
                cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]


            iou_pred_matches_all[cls_id] = np.concatenate((iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            iou_pred_scores_all[cls_id] = np.concatenate((iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]
            iou_gt_matches_all[cls_id] = np.concatenate((iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1)

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)

                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                if len(iou_thres_pred_match) > 0 and pred_idx_mapping is not None:
                    pred_idx_mapping = dict([(i, pred_idx_mapping[j]) for i, j in enumerate(np.where(iou_thres_pred_match > -1)[0])])
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4))

                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                
                if len(iou_thres_gt_match) > 0 and gt_idx_mapping is not None:
                    gt_idx_mapping = dict([(i, gt_idx_mapping[j]) for i, j in enumerate(np.where(iou_thres_gt_match > -1)[0])])
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)

            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_RTs, cls_gt_handle_visibility, 
                                              cls_pred_class_ids, cls_pred_RTs,
                                              synset_names)


            pose_cls_gt_match, pose_cls_pred_match = compute_match_from_degree_cm(RT_overlaps, 
                                                                                  cls_pred_class_ids, 
                                                                                  cls_gt_class_ids, 
                                                                                  degree_thres_list, 
                                                                                  shift_thres_list)
            for i in range(pose_cls_pred_match.shape[2]):
                pose_pred_matches[:, :, progress, pred_idx_mapping[i]] = np.vectorize(lambda k: gt_idx_mapping[k] if k != -1 else -1)(pose_cls_pred_match[:, :, i])
            for i in range(pose_cls_gt_match.shape[2]):
                pose_gt_matches[:, :, progress, gt_idx_mapping[i]] = np.vectorize(lambda k: pred_idx_mapping[k] if k != -1 else -1)(pose_cls_gt_match[:, :, i])
            pose_pred_matches_all[cls_id] = np.concatenate((pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1)
            
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            pose_pred_scores_all[cls_id]  = np.concatenate((pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert pose_pred_scores_all[cls_id].shape[2] == pose_pred_matches_all[cls_id].shape[2], '{} vs. {}'.format(pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape)
            pose_gt_matches_all[cls_id] = np.concatenate((pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1)

    
    
    # draw iou 3d AP vs. iou thresholds
    fig_iou = plt.figure()
    ax_iou = plt.subplot(111)
    plt.ylabel('AP')
    plt.ylim((0, 1))
    plt.xlabel('3D IoU thresholds')
    iou_output_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.png'.format(iou_thres_list[0], iou_thres_list[-1]))
    iou_dict_pkl_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.pkl'.format(iou_thres_list[0], iou_thres_list[-1]))

    iou_dict = {}
    iou_dict['thres_list'] = iou_thres_list
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        # print(class_name)
        for s, iou_thres in enumerate(iou_thres_list):
            iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
                                                                   iou_pred_scores_all[cls_id][s, :],
                                                                   iou_gt_matches_all[cls_id][s, :])    
        ax_iou.plot(iou_thres_list, iou_3d_aps[cls_id, :], label=class_name)
        
    iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-1, :], axis=0)
    ax_iou.plot(iou_thres_list, iou_3d_aps[-1, :], label='mean')
    ax_iou.legend()
    fig_iou.savefig(iou_output_path)
    plt.close(fig_iou)

    iou_dict['aps'] = iou_3d_aps
    with open(iou_dict_pkl_path, 'wb') as f:
        pickle.dump(iou_dict, f)
    

    # draw pose AP vs. thresholds
    if use_matches_for_pose:
        prefix='Pose_Only_'
    else:
        prefix='Pose_Detection_'


    pose_dict_pkl_path = os.path.join(log_dir, prefix+'AP_{}-{}degree_{}-{}cm.pkl'.format(degree_thres_list[0], degree_thres_list[-2], 
                                                                                          shift_thres_list[0], shift_thres_list[-2]))
    pose_dict = {}
    pose_dict['degree_thres'] = degree_thres_list
    pose_dict['shift_thres_list'] = shift_thres_list

    for i, degree_thres in enumerate(degree_thres_list):                
        for j, shift_thres in enumerate(shift_thres_list):
            # print(i, j)
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

                pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(cls_pose_pred_matches_all, 
                                                                        cls_pose_pred_scores_all, 
                                                                        cls_pose_gt_matches_all)

            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
    
    pose_dict['aps'] = pose_aps
    with open(pose_dict_pkl_path, 'wb') as f:
        pickle.dump(pose_dict, f)


    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        # print(class_name)
        # print(np.amin(aps[i, :, :]), np.amax(aps[i, :, :]))
    
        #ap_image = cv2.resize(pose_aps[cls_id, :, :]*255, (320, 320), interpolation = cv2.INTER_LINEAR)
        fig_iou = plt.figure()
        ax_iou = plt.subplot(111)
        plt.ylabel('Rotation thresholds/degree')
        # plt.ylim((degree_thres_list[0], degree_thres_list[-2]))
        plt.xlabel('translation/cm')
        # plt.xlim((shift_thres_list[0], shift_thres_list[-2]))
        plt.imshow(pose_aps[cls_id, :-1, :-1][::-1], cmap='jet', interpolation='bilinear', extent=[shift_thres_list[0], shift_thres_list[-2], degree_thres_list[0], degree_thres_list[-2]])

        output_path = os.path.join(log_dir, prefix+'AP_{}_{}-{}degree_{}-{}cm.png'.format(class_name, 
                                                                                   degree_thres_list[0], degree_thres_list[-2], 
                                                                                   shift_thres_list[0], shift_thres_list[-2]))
        plt.colorbar()
        plt.savefig(output_path)
        plt.close(fig_iou)
    
    #ap_mean_image = cv2.resize(pose_aps[-1, :, :]*255, (320, 320), interpolation = cv2.INTER_LINEAR) 
    
    fig_pose = plt.figure()
    ax_pose = plt.subplot(111)
    plt.ylabel('Rotation thresholds/degree')
    # plt.ylim((degree_thres_list[0], degree_thres_list[-2]))
    plt.xlabel('translation/cm')
    # plt.xlim((shift_thres_list[0], shift_thres_list[-2]))
    plt.imshow(pose_aps[-1, :-1, :-1][::-1], cmap='jet', interpolation='bilinear', extent=[shift_thres_list[0], shift_thres_list[-2], degree_thres_list[0], degree_thres_list[-2]])
    output_path = os.path.join(log_dir, prefix+'mAP_{}-{}degree_{}-{}cm.png'.format(degree_thres_list[0], degree_thres_list[-2], 
                                                                             shift_thres_list[0], shift_thres_list[-2]))
    plt.colorbar()
    plt.savefig(output_path)
    plt.close(fig_pose)

    
    fig_rot = plt.figure()
    ax_rot = plt.subplot(111)
    plt.ylabel('AP')
    plt.ylim((0, 1.05))
    plt.xlabel('translation/cm')
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        # print(class_name)
        ax_rot.plot(shift_thres_list[:-1], pose_aps[cls_id, -1, :-1], label=class_name)
    
    ax_rot.plot(shift_thres_list[:-1], pose_aps[-1, -1, :-1], label='mean')
    output_path = os.path.join(log_dir, prefix+'mAP_{}-{}cm.png'.format(shift_thres_list[0], shift_thres_list[-2]))
    ax_rot.legend()
    fig_rot.savefig(output_path)
    plt.close(fig_rot)

    fig_trans = plt.figure()
    ax_trans = plt.subplot(111)
    plt.ylabel('AP')
    plt.ylim((0, 1.05))

    plt.xlabel('Rotation/degree')
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        # print(class_name)
        ax_trans.plot(degree_thres_list[:-1], pose_aps[cls_id, :-1, -1], label=class_name)

    ax_trans.plot(degree_thres_list[:-1], pose_aps[-1, :-1, -1], label='mean')
    output_path = os.path.join(log_dir, prefix+'mAP_{}-{}degree.png'.format(degree_thres_list[0], degree_thres_list[-2]))
    
    ax_trans.legend()
    fig_trans.savefig(output_path)
    plt.close(fig_trans)

    iou_aps = iou_3d_aps
    for cls_id in range(1, num_classes):
        print('{} 3D IoU at 25: {:.1f}'.format(synset_names[cls_id], iou_aps[cls_id, iou_thres_list.index(0.25)] * 100))
        print('{} 3D IoU at 50: {:.1f}'.format(synset_names[cls_id], iou_aps[cls_id, iou_thres_list.index(0.5)] * 100))
        
    print('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.25)] * 100))
    print('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.5)] * 100))
    
    for cls_id in range(1, num_classes):
        for deg_thresh in degree_thres_list:
            for shift_thres in shift_thres_list:
                print('{} {} degree, {}cm: {:.1f}'.format(synset_names[cls_id], deg_thresh, shift_thres, pose_aps[cls_id, degree_thres_list.index(deg_thresh), shift_thres_list.index(shift_thres)] * 100))
    
    for deg_thresh in degree_thres_list:
        for shift_thres in shift_thres_list:
            print('{} degree, {}cm: {:.1f}'.format(deg_thresh, shift_thres, pose_aps[-1, degree_thres_list.index(deg_thresh), shift_thres_list.index(shift_thres)] * 100))

    return iou_3d_aps, pose_aps, pose_pred_matches, pose_gt_matches

