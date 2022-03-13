import cupy as cp
import numpy as np
    
ppf_kernel = cp.RawKernel(r'''
    #include "helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void ppf_voting(
        const float *points, const float *outputs, const float *probs, const int *point_idxs, float *grid_obj, const float *corner, const float res,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            float prob = max(probs[a_idx], probs[b_idx]);
            float3 co = make_float3(0.f, -ab.z, ab.y);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 center_grid = (c + offset - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 || 
                    center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                    continue;
                }
                int3 center_grid_floor = make_int3(center_grid);
                int3 center_grid_ceil = center_grid_floor + 1;
                float3 residual = fracf(center_grid);
                
                float3 w0 = 1.f - residual;
                float3 w1 = residual;
                
                float lll = w0.x * w0.y * w0.z;
                float llh = w0.x * w0.y * w1.z;
                float lhl = w0.x * w1.y * w0.z;
                float lhh = w0.x * w1.y * w1.z;
                float hll = w1.x * w0.y * w0.z;
                float hlh = w1.x * w0.y * w1.z;
                float hhl = w1.x * w1.y * w0.z;
                float hhh = w1.x * w1.y * w1.z;

                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], lll * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], llh * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], lhl * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], lhh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], hll * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], hlh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], hhl * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], hhh * prob);
            }
        }
    }
''', 'ppf_voting', options=('-I models/include',))


backvote_kernel = cp.RawKernel(r'''
    #include "helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void backvote(
        const float *points, const float *outputs, float3 *out_offsets, const int *point_idxs, const float *corner, const float res,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z, const float *gt_center, const float tol
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            float3 co = make_float3(0.f, -ab.z, ab.y);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            
            out_offsets[idx] = make_float3(0, 0, 0);
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 pred_center = c + offset;
                if (length(pred_center - make_float3(gt_center[0], gt_center[1], gt_center[2])) > tol) continue;
                float3 center_grid = (pred_center - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0 || center_grid.y < 0 || center_grid.z < 0 || 
                    center_grid.x >= grid_x - 1 || center_grid.y >= grid_y - 1 || center_grid.z >= grid_z - 1) {
                    continue;
                }
                out_offsets[idx] = -offset;
                break;
            }
        }
    }
''', 'backvote', options=('-I models/include',))

rot_voting_kernel = cp.RawKernel(r'''
    #include "helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void rot_voting(
        const float *points, const float *outputs, const float *outputs_rot, float3 *outputs_up, const int *point_idxs, const float *corner, const float res,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            float rot = outputs_rot[idx];
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            float3 co = make_float3(0.f, -ab.z, ab.y);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            
            for (int i = 0; i < n_rots; i++) {
                float angle = i * 2 * M_PI / n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 ax = offset / (length(offset) + 1e-7);
                float3 up = cos(rot) * ab + sin(rot) * ax;
                outputs_up[idx * n_rots + i] = up;
            }
        }
    }
''', 'rot_voting', options=('-I models/include',))

findpeak_kernel = cp.RawKernel(r'''
    #include "models/src/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void findpeak(
        const float *grids, float *outputs, int width, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < grid_x * grid_y * grid_z) {
            int x = idx / (grid_y * grid_z);
            int yz = idx % (grid_y * grid_z);
            int y = yz / grid_z;
            int z = yz % grid_z;
            float diff_x = grids[idx] - grids[min(grid_x - 1, x + width) * grid_y * grid_z + y * grid_z + z]
                           + grids[idx] - grids[max(0, x - width) * grid_y * grid_z + y * grid_z + z];
            float diff_y = grids[idx] - grids[x * grid_y * grid_z, min(grid_y - 1, y + width) * grid_z + z]
                           + grids[idx] - grids[x * grid_y * grid_z, max(0, y - width) * grid_z + z];
            float diff_z = grids[idx] - grids[x * grid_y * grid_z + y * grid_z + min(grid_z - 1, z + width)]
                           + grids[idx] - grids[x * grid_y * grid_z + y * grid_z + max(0, z - width)];
            outputs[idx] = diff_x + diff_y + diff_z;
        }
    }
''', 'findpeak', options=('-I models/include',))
