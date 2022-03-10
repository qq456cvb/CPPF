import torch
import os
import numpy as np
from glob import glob
import cv2
from .util import real2prob, backproject, rotx, roty
import open3d as o3d
import trimesh
import hydra
import MinkowskiEngine as ME
from scipy.spatial.transform import Rotation as R
import OpenEXR
    

def generate_target(pc, pc_normal, up_sym=False, right_sym=False, z_right=False, subsample=200000):
    if subsample is None:
        xv, yv = np.meshgrid(np.arange(pc.shape[1]), np.arange(pc.shape[1]))
        point_idxs = np.stack([yv, xv], -1).reshape(-1, 2)
    else:
        point_idxs = np.random.randint(0, pc.shape[0], size=[subsample, 2])
                
    a = pc[point_idxs[:, 0]]
    b = pc[point_idxs[:, 1]]
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    proj_len = np.sum(a * pdist_unit, -1)
    oc = a - proj_len[..., None] * pdist_unit
    dist2o = np.linalg.norm(oc, axis=-1)
    # print(proj_len.shape, dist2o.shape)
    # print(proj_len.min(), proj_len.max())
    target_tr = np.stack([proj_len, dist2o], -1)

    up = np.array([0, 1, 0])
    down = np.array([0, -1, 0])
    if z_right:
        right = np.array([0, 0, 1])
        left = np.array([0, 0, -1])
    else:
        right = np.array([1, 0, 0])
        left = np.array([-1, 0, 0])
    up_cos = np.arccos(np.sum(pdist_unit * up, -1))
    if up_sym:
        up_cos = np.minimum(up_cos, np.arccos(np.sum(pdist_unit * down, -1)))
    right_cos = np.arccos(np.sum(pdist_unit * right, -1))
    if right_sym:
        right_cos = np.minimum(right_cos, np.arccos(np.sum(pdist_unit * left, -1)))
    target_rot = np.stack([up_cos, right_cos], -1)
    
    pairwise_normals = pc_normal[point_idxs[:, 0]]
    pairwise_normals[np.sum(pairwise_normals * pdist_unit, -1) < 0] *= -1
    target_rot_aux = np.stack([
        np.sum(pairwise_normals * up, -1) > 0,
        np.sum(pairwise_normals * right, -1) > 0
    ], -1).astype(np.float32)
    return target_tr.astype(np.float32).reshape(-1, 2), target_rot.astype(np.float32).reshape(-1, 2), target_rot_aux.reshape(-1, 2), point_idxs.astype(np.int64)


def augment_rgb(rgb, bg_color):
    fg_mask = np.any(rgb != bg_color, -1)
    rgb[fg_mask] *= (1 + 0.4 * np.random.random(3) - 0.2) # brightness change for each channel
    rgb[fg_mask] += np.expand_dims((0.05 * np.random.random(rgb[fg_mask].shape[:-1]) - 0.025), -1) # jittering on each pixel
    rgb[fg_mask] = np.clip(rgb[fg_mask], 0, 1)
    return rgb


# IMPORTANT: nocs coord, 
#      |y
#      |
#      |
# z----\   
#       \
#        \x
#
# shapenet coord,
#      |y
#   z\ |
#     \|
# x----\  
class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, model_names):
        super().__init__()
        os.environ.update(
            OMP_NUM_THREADS = '1',
            OPENBLAS_NUM_THREADS = '1',
            NUMEXPR_NUM_THREADS = '1',
            MKL_NUM_THREADS = '1',
            PYOPENGL_PLATFORM = 'osmesa',
            PYOPENGL_FULL_LOGGING = '0'
        )
        self.cfg = cfg
        self.intrinsics = np.array([[591.0125, 0, 320], [0, 590.16775, 240], [0, 0, 1]])
        
        self.model_names = []
        for name in model_names:
            self.model_names.append(name)
        self.is_nocs = cfg.category in ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
            
    def get_item_impl(self, model_name):
        import OpenGL
        OpenGL.FULL_LOGGING = False
        OpenGL.ERROR_LOGGING = False
        from pyrender import DirectionalLight, SpotLight, Mesh, Node, Scene, OffscreenRenderer, RenderFlags, Camera
        class PinholeCamera(Camera):
            def __init__(self, fx, fy, w, h):
                super().__init__(zfar=None)
                self.fx = fx
                self.fy = fy
                self.w = w
                self.h = h
                
            @property
            def zfar(self):
                return self._zfar

            @zfar.setter
            def zfar(self, value):
                if value is not None:
                    value = float(value)
                    if value <= 0 or value <= self.znear:
                        raise ValueError('zfar must be >0 and >znear')
                self._zfar = value

            def get_projection_matrix(self, width=None, height=None):
                n = self.znear
                f = self.zfar

                P = np.zeros((4,4))
                P[0][0] = 1 / self.w * 2 * self.fx
                P[1][1] = 1 / self.h * 2 * self.fy
                P[3][2] = -1.0

                if f is None:
                    P[2][2] = -1.0
                    P[2][3] = -2.0 * n
                else:
                    P[2][2] = (f + n) / (n - f)
                    P[2][3] = (2 * f * n) / (n - f)

                return P
            
        r = OffscreenRenderer(viewport_width=640, viewport_height=480)
        shapenet_cls, mesh_name = model_name.split('/')
        path = os.path.join(hydra.utils.to_absolute_path(self.cfg.shapenet_root), f'{shapenet_cls}/{mesh_name}/models/model_normalized.obj')
        mesh = trimesh.load(path)
        obj_scale = self.cfg.scale_range
        
        mesh_pose = np.eye(4)
        if self.is_nocs:
            # rot
            y_angle = np.random.uniform(0, 2 * np.pi)
            x_angle = np.random.uniform(25 / 180 * np.pi, 65 / 180 * np.pi)
            yy_angle = np.random.uniform(-15 / 180 * np.pi, 15 / 180 * np.pi)
            mesh_pose[:3, :3] =  roty(yy_angle)[:3, :3] @ rotx(x_angle)[:3, :3] @ roty(y_angle)[:3, :3]
            
            # tr
            tr = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), -np.random.uniform(0.6, 2.0)])
            mesh_pose[:3, -1] = tr
        else:
            # rot
            y_angle = np.random.uniform(0, 2 * np.pi)
            x_angle = np.clip(np.random.normal(40, 10), 10, 70) / 180 * np.pi
            mesh_pose[:3, :3] = rotx(x_angle)[:3, :3] @ roty(y_angle)[:3, :3]
            
            # tr
            tr = np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), -np.random.uniform(1.0, 5.0)])
            mesh_pose[:3, -1] = tr

        
        bounds = mesh.bounds
        trans_mat = np.eye(4)
        trans_mat[:3, -1] = -(bounds[1] + bounds[0]) / 2
        
        scale_mat = np.eye(4)
        scale = np.random.uniform(obj_scale[0], obj_scale[1])
        scale_mat[:3, :3] *= scale
        mesh.apply_transform(mesh_pose @ scale_mat @ trans_mat)
        if isinstance(mesh, trimesh.Scene):
            scene = Scene.from_trimesh_scene(mesh)
            scene.bg_color = np.random.rand(3)
        else:
            scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]), bg_color=np.random.rand(3))
            scene.add(Mesh.from_trimesh(mesh), pose=np.eye(4))
        
        direc_l = DirectionalLight(color=np.ones(3), intensity=np.random.uniform(5, 15))
        spot_l = SpotLight(color=np.ones(3), intensity=np.random.uniform(0, 10),
                        innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)

        cam_pose = np.eye(4)
        cam = PinholeCamera(591.0125, 590.16775, 640, 480)
        
        scene.add(cam, pose=cam_pose)
        scene.add(direc_l, pose=cam_pose)
        scene.add(spot_l, pose=cam_pose)
        
        depth = r.render(scene, flags=RenderFlags.DEPTH_ONLY)
        r.delete()
        
        mask = (depth > 0).astype(bool)
        
        pc, _ = backproject(depth, self.intrinsics, mask)
        pc[:, 0] = -pc[:, 0]
        pc[:, 2] = -pc[:, 2]
        pc -= tr
        pc = (np.linalg.inv(mesh_pose[:3, :3]) @ pc.T).T
        if self.is_nocs:
            # rotate to nocs coord
            flip2nocs = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            pc = (flip2nocs @ pc.T).T
        
        # random jitter, all point together
        pc = pc + np.clip(self.cfg.res / 4 * np.random.randn(*pc.shape), -self.cfg.res / 2, self.cfg.res / 2)
        
        discrete_coords, indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=self.cfg.res)
        pc = pc[indices]
        
        if pc.shape[0] < 100 or pc.shape[0] > self.cfg.npoint_max:
            return self.get_item_impl(self.model_names[np.random.randint(len(self))])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.cfg.knn))
        normals = np.array(pcd.normals).astype(np.float32)
        
        # dummy normal
        targets_tr, targets_rot, targets_rot_aux, point_idxs = generate_target(pc, normals, self.cfg.up_sym, self.cfg.right_sym, self.cfg.z_right, 200000)
        
        if self.cfg.cls_bins:
            vote_range = self.cfg.vote_range
            targets_tr = np.stack([
                real2prob(np.clip(targets_tr[:, 0] + vote_range[0], 0, 2 * vote_range[0]), 2 * vote_range[0], self.cfg.tr_num_bins, circular=False),
                real2prob(np.clip(targets_tr[:, 1], 0, vote_range[1]), vote_range[1], self.cfg.tr_num_bins, circular=False),
            ], 1)
        
        if self.cfg.cls_bins:
            targets_rot = np.stack([
                real2prob(targets_rot[:, 0], np.pi, self.cfg.rot_num_bins, circular=False),
                real2prob(targets_rot[:, 1], np.pi, self.cfg.rot_num_bins, circular=False),
            ], 1)
                
        targets_scale = np.log(((bounds[1] - bounds[0]) / 2).astype(np.float32) * scale) - np.log(np.array(self.cfg.scale_mean))
        # print(targets_scale)
        # return pc_colors, (depth / 1000).astype(np.float32), np.stack(idxs, -1).astype(np.int64), \
        return pc.astype(np.float32), normals, targets_tr, targets_rot, targets_rot_aux, targets_scale.astype(np.float32), point_idxs
        
    def __getitem__(self, idx):
        model_name = self.model_names[idx]
        return self.get_item_impl(model_name)
    
    def __len__(self):
        return min(len(self.model_names), 200)
    

class BlenderLaptopAuxDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, model_names):
        super().__init__()
        self.cfg = cfg
        self.intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        
        self.model_names = []
        blacklists = open(hydra.utils.to_absolute_path('data/laptop_nonreal.txt')).read().splitlines()
        for name in model_names:
            if name not in blacklists:
                self.model_names.append(name)
            
        self.camera_lens = 29.55062484741211
        self.camera_sensor_width = 32.
        self.resolution_x = 640
        self.resolution_y = 480
    
    def __len__(self):
        return len(self.model_names)
    
    def backproject(self, depth, mask):
        sensor_height = self.resolution_y / self.resolution_x * self.camera_sensor_width
        u, v = np.meshgrid(np.arange(self.resolution_x), np.arange(self.resolution_y))
        
        u = u[mask]
        v = v[mask]
        
        x = (0.5 - u / self.resolution_x) * self.camera_sensor_width / self.camera_lens
        y = (0.5 - v / self.resolution_y) * sensor_height / self.camera_lens
        z = np.ones_like(x)
        
        norm = np.linalg.norm(np.stack([x, y, z], -1), axis=-1)
        
        u = (0.5 - x * self.camera_lens / self.camera_sensor_width) * self.resolution_x
        v = (0.5 - y * self.camera_lens / sensor_height) * self.resolution_y
        
        intrinsics_inv = np.linalg.inv(self.intrinsics)
        
        grid = np.stack([u, v], -1).T

        # shape: height * width
        # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
        # mesh_grid = np.reshape(mesh_grid, [2, -1])
        length = grid.shape[1]
        ones = np.ones([1, length])
        uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

        xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
        xyz = np.transpose(xyz) #[num_pixel, 3]

        z = depth[mask] / norm

        # print(np.amax(z), np.amin(z))
        pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]

        return pts
    
    def __getitem__(self, idx):
        shapenet_cls, mesh_name = self.model_names[idx].split('/')
        # print(shapenet_cls, mesh_name)
        img_idx = np.random.randint(1, 21)
        img_path = hydra.utils.to_absolute_path(os.path.join(self.cfg.data_root, f'{shapenet_cls}/{mesh_name}/{img_idx}.png'))
        depth_path = hydra.utils.to_absolute_path(os.path.join(self.cfg.data_root, f'{shapenet_cls}/{mesh_name}/{img_idx}_depth0001.exr'))
        if not os.path.exists(img_path):
            return self[np.random.randint(len(self))]
        
        tr = np.load(img_path.replace('.png', '.tr.npy'))
        rot = np.load(img_path.replace('.png', '.rot.npy'))
        scale = np.load(img_path.replace('.png', '.scale.npy'))
        
        beta = R.from_matrix(rot).as_euler('yxy', degrees=True)
        # print(beta)
        if beta[1] > 60 or np.abs(tr[2]) < 0.8:
            return self[np.random.randint(len(self))]
        
        rgb = cv2.imread(img_path)[:, :, ::-1]
        depth = np.frombuffer(OpenEXR.InputFile(depth_path).channel('R'), np.float32).reshape(480, 640).copy()
        depth[depth > 100] = 0
        rgb[depth == 0] = np.full((3,), 255)
        
        mask = (depth > 0).astype(bool)
        if np.sum(mask) < 100:
            return self[np.random.randint(len(self))]
        
        idxs = np.where(mask)
        
        pc = self.backproject(depth, mask)
        pc[:, 0] = -pc[:, 0]
        pc[:, 2] = -pc[:, 2]
        
        pc += tr
        pc = (rot.T @ pc.T).T
        
        # crop image
        bbox = np.array([
            [np.min(idxs[0]), np.max(idxs[0])],
            [np.min(idxs[1]), np.max(idxs[1])]
        ])
        rgb = (cv2.resize(rgb[bbox[0][0]:bbox[0][1]+1, bbox[1][0]:bbox[1][1]+1], (224, 224)) / 255.).astype(np.float32)
        depth = cv2.resize(depth[bbox[0][0]:bbox[0][1]+1, bbox[1][0]:bbox[1][1]+1], (224, 224), interpolation=cv2.INTER_NEAREST)
        
        label = np.full((rgb.shape[0], rgb.shape[1]), -100, np.int64)
        
        resize_scale = 224 / (bbox[:, 1] - bbox[:, 0])
        pc_xy = np.stack(idxs, -1)
        idxs_resized = np.clip(((pc_xy - bbox[:, 0]) * resize_scale).astype(np.int64), 0, 223)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        plane1, inlier1 = pcd.segment_plane(distance_threshold=0.01,
                                    ransac_n=3,
                                    num_iterations=100)
        if len(inlier1) > pc.shape[0] - 3:
            return self[np.random.randint(len(self))]
        idx_mapping = np.delete(np.arange(pc.shape[0]), inlier1)
        pcd.points = o3d.utility.Vector3dVector(np.delete(pc, inlier1, axis=0))
        plane2, inlier2 = pcd.segment_plane(distance_threshold=0.01,
                                    ransac_n=3,
                                    num_iterations=100)
        inlier2 = idx_mapping[inlier2]
        base = inlier1
        screen = inlier2
        
        if np.abs(plane1[1]) < np.abs(plane2[1]):
            base = inlier2
            screen = inlier1
            
        label[idxs_resized[base][:, 0], idxs_resized[base][:, 1]] = 0
        label[idxs_resized[screen][:, 0], idxs_resized[screen][:, 1]] = 1
        rgb = augment_rgb(rgb, np.full((3,), 1.))
        return rgb, label

