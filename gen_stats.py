import argparse
from utils.dataset import generate_target
from utils.util import typename2shapenetid
import os
import open3d as o3d
import numpy as np
from tqdm import tqdm

from utils.util import estimate_normals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bowl', help='Category name')
    parser.add_argument('--shapenet_root', default='/home/neil/disk/ShapeNetCore.v2',  help='ShapeNet root directory')
    parser.add_argument('--up_sym', action='store_true',  help='If the objects look similar to a cylinder from up ([0, 1, 0]) to bottom')
    parser.add_argument('--right_sym', action='store_true',  help='If the objects look similar to a cylinder from left to right')
    parser.add_argument('--z_right', action='store_true',  help='If the objects use [0, 0, 1] as the right axis (default, [1, 0, 0])')
    args = parser.parse_args()
    
    name_path = 'data/shapenet_names/{}.txt'.format(args.category)
    if os.path.exists(name_path):
        shapenames = open(name_path).read().splitlines()
    else:
        shapenet_id = typename2shapenetid[args.category]
        shapenames = os.listdir(os.path.join(args.shapenet_root, '{}'.format(shapenet_id)))
        shapenames = [shapenet_id + '/' + name for name in shapenames]
    
    scale_range = [np.inf, -np.inf]
    vote_range = [0, 0]
    scale_mean = []
    for model_name in tqdm(shapenames):
        shapenet_cls, mesh_name = model_name.split('/')
        path = os.path.join(args.shapenet_root, f'{shapenet_cls}/{mesh_name}/models/model_normalized.obj')
        mesh = o3d.io.read_triangle_mesh(path)
        pc = np.array(mesh.sample_points_uniformly(2048).points)
        
        # normalize to center
        pc -= (np.min(pc, 0) + np.max(pc, 0)) / 2
        
        normals = estimate_normals(pc, 60)
        targets_tr = generate_target(pc, normals, args.up_sym, args.right_sym, args.z_right, 100000)[0]
        
        diag_length = np.linalg.norm(np.max(pc, 0) - np.min(pc, 0))
        
        scale_range[0] = min(scale_range[0], diag_length)
        scale_range[1] = max(scale_range[1], diag_length)
        
        vote_range[0] = max(vote_range[0], np.max(np.abs(targets_tr[:, 0])))
        vote_range[1] = max(vote_range[1], np.max(targets_tr[:, 1]))
        
        scale_mean.append(np.max(pc, 0))
    scale_mean = np.mean(scale_mean, 0)
    
    print(f'scale_range: {scale_range}')
    print(f'vote_range: {vote_range}')
    print(f'scale_mean: {scale_mean}')
        