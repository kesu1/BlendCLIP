'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

import random

import torch
import numpy as np
import torch.utils.data as data

import yaml
from easydict import EasyDict

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
from tqdm import tqdm
import pickle
from PIL import Image
import math
import matplotlib.pyplot as plt
#import glob
#from pytorch3d.structures import Pointclouds
#from pytorch3d.ops import estimate_pointcloud_normals

#import open3d as o3d

#import numpy as np
from scipy.spatial import ConvexHull

def hidden_point_removal(points: np.ndarray,
                          camera: np.ndarray,
                          radius: float | None = None,
                          *,
                          tol: float = 1e-12) -> np.ndarray:
    """
    Visibility test for an unorganised 3-D point cloud using the HPR operator.
    
    Parameters
    ----------
    points : (N, 3) ndarray
        Cartesian coordinates of the point cloud.
    camera : (3,) ndarray
        Camera / view-point position in the same coordinate frame.
    radius : float or None, optional
        Radius R of the inversion sphere centred at `camera`.
        If None or smaller than the farthest point, R = f * max ||p-camera||
        with f = 1.1 (a gentle enlargement).  Larger R → more points marked
        visible; smaller R → fewer.
    tol : float, optional
        Numerical tolerance that avoids division by zero for points that
        coincide with the camera.
    
    Returns
    -------
    mask : (N,) bool ndarray
        `True` for points classified as visible, `False` otherwise.
    """
    p = np.ascontiguousarray(points, dtype=float)        # (N,3)
    c = np.asarray(camera,  dtype=float).reshape(1, 3)   # (1,3)
    q = p - c                                           # translate so that camera is at origin

    # Euclidean norms
    r = np.linalg.norm(q, axis=1)                       # (N,)

    # Robust radius choice
    R = float(radius) if radius and radius > r.max() else 1.1 * r.max()
    Rvec = R - r                                        # (N,)

    # Avoid divisions by zero (points exactly at camera)
    denom = np.where(r < tol, np.inf, r)

    # spherical flipping
    flipped = q + (2 * Rvec / denom)[:, None] * q       # (N,3)

    # Add the camera itself (origin) so rear points do not enter the hull
    all_pts = np.vstack((flipped, np.zeros((1, 3), dtype=float)))

    hull = ConvexHull(all_pts, qhull_options='QJ')  # triangulate the points
    hull_verts = np.unique(hull.vertices)               # indices into all_pts

    # Last point in all_pts is the origin we appended
    visible_idx = hull_verts[hull_verts < len(points)]  # discard origin

    mask = np.zeros(len(points), dtype=bool)
    mask[visible_idx] = True
    return mask


def simulate_occlusion(points, param):    
    """Occlusion simulation using the HPR operator."""
    
    if points.shape[0] == 0:
        raise ValueError("Input points array is empty.")
    
    try:
        original_points = points.copy()
        
        scale = 10**param
        
        # Camera positioning with limits
        base_radius = 3
        camera_dist = base_radius + np.random.uniform(-0.5, 0.5)
        
        phi = np.arccos(1 - 2 * np.random.rand()) 
        theta = np.random.uniform(0, 2 * np.pi)
        
        camera_location = np.array([
            camera_dist * np.sin(phi) * np.cos(theta),
            camera_dist * np.sin(phi) * np.sin(theta),
            camera_dist * np.cos(phi)
        ])
        
        # Calculate HPR radius with safety limits
        max_dist = np.max(np.linalg.norm(points[:, :3] - camera_location, axis=1))
        hpr_radius = max_dist * scale
        
        # Safety check for radius
        if hpr_radius <= 0 or np.isinf(hpr_radius) or np.isnan(hpr_radius):
            raise ValueError("Invalid HPR radius calculated.")
            
        pt_map = hidden_point_removal(points[:, :3], camera_location, radius=hpr_radius)
        
        if len(pt_map) == 0:
            return original_points
        
        return points[pt_map, :], camera_location
            
    except Exception as e:
        # On any error, return the original points
        print(f"Warning: Error in simulate_occlusion: {e}, returning original points")
        return original_points

def get_normal_params(n_points):
    """
    Choose (radius, max_nn) for Open3D hybrid normal estimation  
    on unit‑sphere normalized, occluded LiDAR object clouds.
    """
    if n_points >= 9999: # hardcoded for objaverse dense
        return 0.05, 30
    if n_points < 512:
        # very sparse / skinny -> need ~30% of object diameter
        return 0.3, 20
    elif n_points < 1024:
        # mid‑small -> ~20% support
        return 0.2, 30
    elif n_points < 2048:
        # moderate -> ~15% support
        return 0.15, 30
    else:
        # dense -> ~10% support
        return 0.10, 30


def estimate_normals_open3d(points, orient=True):
    """Estimate normals with consistent orientation toward the object center."""
    import open3d as o3d
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    radius, max_nn = get_normal_params(points.shape[0])

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )

    # Orient normals
    if orient:
        #center = np.mean(points, axis=0)
        #pcd.orient_normals_towards_camera_location(center)
        pcd.orient_normals_to_align_with_direction()

    return np.asarray(pcd.normals)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pc_normalize(pc):
    if pc.shape[0] == 0:  # Check for empty point cloud
        return pc  # Return empty array as-is
    
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    
    if m == 0:  # Handle case where all points are identical (or single point)
        return pc  # Return zero-centered points as-is
    
    pc = pc / m
    return pc

import numba

@numba.jit(nopython=True, fastmath=True)
def _fps_numba(xyz, npoint, start_idx):
    """Numba-compiled FPS core loop"""
    N = xyz.shape[0]
    centroids = np.zeros(npoint, dtype=np.int32)
    distances = np.full(N, np.inf, dtype=np.float32)
    
    farthest = start_idx
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        
        # Update distances
        for j in range(N):
            dist = 0.0
            for k in range(3):
                diff = xyz[j, k] - centroid[k]
                dist += diff * diff
            if dist < distances[j]:
                distances[j] = dist
        
        # Find farthest point
        max_dist = -1.0
        for j in range(N):
            if distances[j] > max_dist:
                max_dist = distances[j]
                farthest = j
                
    return centroids

def farthest_point_sample(point, npoint, seed=None):
    """
    Numba-accelerated FPS implementation
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
        seed: random seed for reproducibility (optional)
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3].astype(np.float32)
    
    if seed:
        local_rng = np.random.Generator(np.random.PCG64(seed))
        start_idx = local_rng.integers(0, N)
    else:
        start_idx = np.random.randint(0, N)
    
    centroids = _fps_numba(xyz, npoint, start_idx)
    return point[centroids]

"""
def farthest_point_sample(point, npoint, seed=None):

    #Input:
    #    xyz: pointcloud data, [N, D]
    #    npoint: number of samples
    #    seed: random seed for reproducibility (optional)
    #Return:
    #    centroids: sampled pointcloud index, [npoint, D]

    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    
    if seed:
        local_rng = np.random.Generator(np.random.PCG64(seed))
        farthest = local_rng.integers(0, N)
    else:
        farthest = np.random.randint(0, N)
        
    #farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
"""

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNxC array, original batch of point clouds
        Return:
          BxNxC array, rotated batch of point clouds (XYZ rotated, others preserved)
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        
        # Rotate only XYZ coordinates
        rotated_xyz = np.dot(shape_pc[:, :3], rotation_matrix)
        
        # Assign rotated XYZ and original other features (RGB, etc.)
        rotated_data[k, :, :3] = rotated_xyz
        if shape_pc.shape[1] > 3: # Check if there are features beyond XYZ
             rotated_data[k, :, 3:] = shape_pc[:, 3:]
             
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNxC '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            # Replace dropped points' features with infinity
            #batch_pc[b, drop_idx, :] = np.inf
            batch_pc[b, drop_idx, :] = batch_pc[b,0,:]
    
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNxC array, original batch of point clouds
        Return:
            BxNxC array, scaled batch of point clouds (only XYZ scaled)
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        # Only apply scale to the first 3 columns (XYZ coordinates)
        batch_data[batch_index, :, :3] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNxC array, original batch of point clouds
        Return:
          BxNxC array, shifted batch of point clouds (only XYZ shifted)
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        # Only apply shift to the first 3 columns (XYZ coordinates)
        batch_data[batch_index, :, :3] += shifts[batch_index, :]
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNxC array, original batch of point clouds
        Return:
          BxNxC array, jittered batch of point clouds (only XYZ jittered)
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    # Jitter only the first 3 dimensions (XYZ)
    jitter3d = np.clip(sigma * np.random.randn(B, N, 3), -1*clip, clip)
    batch_data[:, :, :3] += jitter3d # Add jitter only to XYZ
    return batch_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNxC array, original batch of point clouds
        Return:
          BxNxC array, rotated batch of point clouds (XYZ rotated, others preserved)
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        
        # Rotate only XYZ coordinates
        rotated_xyz = np.dot(shape_pc[:, :3], R)
        
        # Assign rotated XYZ and original other features (RGB, etc.)
        rotated_data[k, :, :3] = rotated_xyz
        if shape_pc.shape[1] > 3: # Check if there are features beyond XYZ
            rotated_data[k, :, 3:] = shape_pc[:, 3:]
            
    return rotated_data

import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class ScheduledBatchSampler(data.Sampler):
    """
    Scheduled sampler that yields a fixed total number of batches per epoch.
    The ratio of NuScenes to Objaverse samples per batch ramps up over epochs based on a schedule.
    Samples are drawn randomly *with replacement* from the entire datasets for each batch based on the ratio.
    This is the non-distributed version.

    Args:
        dataset (Objects_Joint): The dataset to sample from.
        batch_size (int): Size of batches.
        total_batches (int): The fixed total number of batches to yield per epoch.
        total_epochs (int): Total number of training epochs.
        warmup_epochs (int): Number of epochs for the warmup phase (0% NuScenes).
        shuffle (bool, optional): If ``True`` (default), shuffle the combined batch indices.
                                   Also influences the random generator seed.
        seed (int, optional): Random seed used for the random number generator. Default: ``0``.
        drop_last (bool): If ``True``, the sampler will drop the last batch if it cannot be fully filled
                          (e.g., if one dataset is empty). Default: ``False``.
                          Note: With fixed total_batches and sampling with replacement, this only
                          affects batches if a dataset is completely empty.
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 total_batches,
                 total_epochs,
                 warmup_epochs,
                 max_lidar_ratio=0.3,
                 static=False,
                 shuffle=True,
                 seed=0,
                 drop_last=False):
        if not isinstance(dataset, Objects_Joint):
            raise TypeError("Dataset must be an instance of Objects_Joint")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(total_batches, int) or total_batches <= 0:
            raise ValueError("total_batches should be a positive integer value, "
                             "but got total_batches={}".format(total_batches))
        if not isinstance(total_epochs, int) or total_epochs <= 0:
             raise ValueError("total_epochs must be a positive integer.")
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
             raise ValueError("warmup_epochs must be a non-negative integer.")
        if warmup_epochs >= total_epochs:
             raise ValueError("warmup_epochs must be less than total_epochs.")
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.batch_size = batch_size
        self.num_batches = total_batches # Fixed number of batches per epoch
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.max_lidar_ratio = max_lidar_ratio
        self.static = static

        # Store indices as NumPy arrays
        self.all_nuscenes_indices = np.array([i for i, data_info in enumerate(dataset.datapath) if data_info[0] == "nuscenes"], dtype=np.int64)
        self.objaverse_indices = np.array([i for i, data_info in enumerate(dataset.datapath) if data_info[0] == "objaverse"], dtype=np.int64)

        if len(self.all_nuscenes_indices) == 0 and self.warmup_epochs < self.total_epochs:
             print("Warning: Dataset contains no NuScenes samples, but schedule expects them after warmup. Ratio will be ignored.")
        if len(self.objaverse_indices) == 0:
             print("Warning: Dataset contains no Objaverse samples. Batches will only contain NuScenes (if available).")

        # Total number of samples available
        self.num_nuscenes_total = len(self.all_nuscenes_indices)
        self.num_objaverse_total = len(self.objaverse_indices)

    def _linear_ratio(self):
        """Calculates the target real-world ratio for the current epoch with a linear ramp to max_lidar_ratio."""
        if self.num_nuscenes_total == 0:
            return 0.0
        if self.num_objaverse_total == 0:
            return 1.0

        # If before warmup ends, ratio is 0
        if self.epoch < self.warmup_epochs:
            return 0.0

        # If warmup is disabled or total epochs is invalid relative to warmup
        if self.total_epochs <= self.warmup_epochs:
            return self.max_lidar_ratio if self.num_objaverse_total > 0 else 1.0

        ramp_duration = self.total_epochs - self.warmup_epochs
        ramp_progress = self.epoch - self.warmup_epochs

        # Calculate the linear ratio: starts at 0.0 after warmup, increases to max_lidar_ratio by total_epochs
        # Ensure progress doesn't exceed duration (e.g., if epoch > total_epochs)
        fraction_complete = min(1.0, ramp_progress / ramp_duration)
        target_ratio = fraction_complete * self.max_lidar_ratio

        # Ensure the ratio does not exceed max_lidar_ratio
        return min(self.max_lidar_ratio, target_ratio)
    
    def _const_ratio(self):
        """Returns a constant target ratio of max_lidar_ratio."""
        if self.num_nuscenes_total == 0:
            return 0.0
        if self.num_objaverse_total == 0:
            return 1.0
        return self.max_lidar_ratio

    def __iter__(self):
        # Create a generator for this epoch, seeded for reproducibility
        g = np.random.Generator(np.random.PCG64(self.seed + self.epoch))

        for i in range(self.num_batches):
            # Calculate target counts based on ratio for the current epoch
            nuscenes_ratio = self._linear_ratio() if not self.static else self._const_ratio()
            target_nuscenes_count = round(nuscenes_ratio * self.batch_size)

            # Ensure counts are within bounds and sum to batch_size
            target_nuscenes_count = max(0, min(target_nuscenes_count, self.batch_size))
            target_objaverse_count = self.batch_size - target_nuscenes_count

            # Determine actual number to sample, considering if datasets are empty
            actual_nuscenes_to_sample = target_nuscenes_count if self.num_nuscenes_total > 0 else 0
            actual_objaverse_to_sample = target_objaverse_count if self.num_objaverse_total > 0 else 0
            
            # for logging
            self.nuscenes_in_batch = actual_nuscenes_to_sample

            # If datasets are empty, counts might not sum to batch_size
            total_sampled = actual_nuscenes_to_sample + actual_objaverse_to_sample

            # Handle drop_last: if we couldn't fill the batch (due to empty dataset), skip
            if total_sampled < self.batch_size and self.drop_last:
                print(f"Warning: Skipping batch {i} due to insufficient samples and drop_last=True.")
                continue # Skip this batch iteration

            # Sample indices using the seeded generator *with replacement*
            nuscenes_batch_indices = np.array([], dtype=np.int64)
            if actual_nuscenes_to_sample > 0:
                nuscenes_batch_indices = g.choice(
                    self.all_nuscenes_indices,
                    size=actual_nuscenes_to_sample,
                    replace=True # Sample with replacement
                )

            objaverse_batch_indices = np.array([], dtype=np.int64)
            if actual_objaverse_to_sample > 0:
                objaverse_batch_indices = g.choice(
                    self.objaverse_indices,
                    size=actual_objaverse_to_sample,
                    replace=True # Sample with replacement
                )

            # Combine indices
            if len(nuscenes_batch_indices) > 0 and len(objaverse_batch_indices) > 0:
                batch_indices_np = np.concatenate((nuscenes_batch_indices, objaverse_batch_indices))
            elif len(nuscenes_batch_indices) > 0:
                batch_indices_np = nuscenes_batch_indices
            elif len(objaverse_batch_indices) > 0:
                batch_indices_np = objaverse_batch_indices
            else:
                batch_indices_np = np.array([], dtype=np.int64)

            # Shuffle the combined batch if required
            if self.shuffle and len(batch_indices_np) > 0:
                g.shuffle(batch_indices_np)

            yield batch_indices_np.tolist()

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. The epoch is used to calculate the
        NuScenes/Objaverse ratio and to seed the random number generator.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __len__(self):
        # Return the fixed number of batches
        return self.num_batches

class BalancedBatchSampler(data.Sampler):
    """
    Samples elements such that each batch aims for a 50/50 split between
    NuScenes and Objaverse samples. It ensures all NuScenes samples are
    iterated through per epoch, while randomly sampling Objaverse samples.
    Mirrors the structure of DistributedBalancedBatchSampler but for single process.

    Args:
        dataset (Objects_Joint): The dataset to sample from.
        batch_size (int): Size of batches.
        shuffle (bool, optional): If ``True`` (default), shuffle the NuScenes indices
            every epoch.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. Default: ``0``.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``. Default: ``False``.
    """
    def __init__(self, dataset, batch_size, shuffle=True, seed=0, drop_last=False):
        if not isinstance(dataset, Objects_Joint):
            raise TypeError("Dataset must be an instance of Objects_Joint")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        # self.dataset = dataset # Not strictly needed after init
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Store indices as NumPy arrays
        self.nuscenes_indices = np.array([i for i, data_info in enumerate(dataset.datapath) if data_info[0] == "nuscenes"], dtype=np.int64)
        self.objaverse_indices = np.array([i for i, data_info in enumerate(dataset.datapath) if data_info[0] == "objaverse"], dtype=np.int64)

        if len(self.nuscenes_indices) == 0:
            raise ValueError("Dataset must contain samples from NuScenes for balanced sampling.")
        if len(self.objaverse_indices) == 0:
            print("Warning: Dataset does not contain Objaverse samples. Batches will only contain NuScenes.")
            self.nuscenes_batch_size = self.batch_size # Fill entire batch with nuscenes
        else:
            # Target NuScenes samples per batch (aim for 50%)
            self.nuscenes_batch_size = self.batch_size // 2
            if self.nuscenes_batch_size == 0 and self.batch_size > 0:
                self.nuscenes_batch_size = 1

        # Calculate number of batches based on iterating through all NuScenes samples
        if self.nuscenes_batch_size > 0:
            num_nuscenes_total = len(self.nuscenes_indices)
            if self.drop_last:
                # Number of batches based on full NuScenes batches
                self.num_batches = num_nuscenes_total // self.nuscenes_batch_size
            else:
                # Number of batches needed to cover all NuScenes
                self.num_batches = math.ceil(num_nuscenes_total / self.nuscenes_batch_size)
        else: # Only objaverse samples
             num_objaverse_total = len(self.objaverse_indices)
             if self.drop_last:
                self.num_batches = num_objaverse_total // self.batch_size
             else:
                self.num_batches = math.ceil(num_objaverse_total / self.batch_size)


    def __iter__(self):
        # Shuffle NuScenes indices at the start of each epoch
        if self.shuffle:
            # Use a generator seeded for reproducibility if needed, but epoch isn't required
            g = np.random.Generator(np.random.PCG64(self.seed + self.epoch))
            indices_shuffled = g.permutation(self.nuscenes_indices)
        else:
            indices_shuffled = self.nuscenes_indices # Use original order

        nuscenes_ptr = 0
        len_nuscenes = len(indices_shuffled)

        for i in range(self.num_batches):
            # Calculate start and end pointers for NuScenes batch
            start = nuscenes_ptr
            end = nuscenes_ptr + self.nuscenes_batch_size
            current_batch_nuscenes = np.array([], dtype=np.int64) # Default empty

            if start < len_nuscenes:
                # Slice the shuffled NumPy array
                current_batch_nuscenes = indices_shuffled[start:min(end, len_nuscenes)]
                nuscenes_ptr = end
            # else: NuScenes exhausted, current_batch_nuscenes remains empty

            # Check for drop_last condition (handled by num_batches calculation)
            # No explicit break needed here due to how num_batches is calculated

            # Determine how many Objaverse samples are needed
            num_objaverse_needed = self.batch_size - len(current_batch_nuscenes)
            current_batch_objaverse = np.array([], dtype=np.int64) # Empty numpy array

            if num_objaverse_needed > 0 and len(self.objaverse_indices) > 0:
                # Sample Objaverse indices using NumPy (with replacement)
                current_batch_objaverse = np.random.choice(
                    self.objaverse_indices,
                    size=num_objaverse_needed,
                    replace=False
                )

            # Combine using NumPy concatenate
            if len(current_batch_nuscenes) > 0 or len(current_batch_objaverse) > 0:
                batch_indices_np = np.concatenate((current_batch_nuscenes, current_batch_objaverse))
                np.random.shuffle(batch_indices_np)
                yield batch_indices_np.tolist()
            # else: continue # Skip if batch ended up empty
            
    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __len__(self):
        return self.num_batches
    
class DistributedScheduledBatchSampler(data.Sampler):
    """
    Distributed-compatible batch sampler that yields a fixed total number of batches per epoch.
    The ratio of NuScenes to Objaverse samples per batch ramps up over epochs based on a schedule.
    Samples are drawn randomly *with replacement* from the entire datasets for each batch based on the ratio.

    Args:
        dataset (Objects_Joint): The dataset to sample from.
        batch_size (int): Per-GPU batch size.
        total_batches (int): The fixed total number of batches to yield per epoch per GPU.
        total_epochs (int): Total number of training epochs.
        warmup_epochs (int): Number of epochs for the warmup phase (0% NuScenes).
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
        shuffle (bool, optional): If ``True`` (default), shuffle the combined batch indices.
                                   Also influences the random generator seed.
        seed (int, optional): Random seed used for the random number generator. Default: ``0``.
        drop_last (bool): If ``True``, the sampler will drop the last batch if it cannot be fully filled
                          (e.g., if one dataset is empty). Default: ``False``.
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 total_batches,
                 total_epochs,
                 warmup_epochs,
                 num_replicas=None,
                 max_lidar_ratio=0.3,
                 static=False,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 drop_last=False):
        # ... (initialization checks remain the same) ...
        if not isinstance(dataset, Objects_Joint):
            raise TypeError("Dataset must be an instance of Objects_Joint")
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(total_batches, int) or total_batches <= 0:
            raise ValueError("total_batches should be a positive integer value, "
                             "but got total_batches={}".format(total_batches))
        if not isinstance(total_epochs, int) or total_epochs <= 0:
             raise ValueError("total_epochs must be a positive integer.")
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
             raise ValueError("warmup_epochs must be a non-negative integer.")
        if warmup_epochs >= total_epochs:
             raise ValueError("warmup_epochs must be less than total_epochs.")
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size # Per-GPU batch size
        self.num_batches = total_batches # Fixed number of batches per epoch
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0 # Track epoch for ratio calculation and shuffling
        self.max_lidar_ratio = max_lidar_ratio
        self.static = static

        # Get all indices globally
        self.all_nuscenes_indices = np.array([i for i, data_info in enumerate(dataset.datapath) if data_info[0] == "nuscenes"], dtype=np.int64)
        self.objaverse_indices = np.array([i for i, data_info in enumerate(dataset.datapath) if data_info[0] == "objaverse"], dtype=np.int64)

        if len(self.all_nuscenes_indices) == 0 and self.warmup_epochs < self.total_epochs:
             print(f"Rank {self.rank} Warning: Dataset contains no NuScenes samples, but schedule expects them after warmup. Ratio will be ignored.")
        if len(self.objaverse_indices) == 0:
             print(f"Rank {self.rank} Warning: Dataset contains no Objaverse samples. Batches will only contain NuScenes (if available).")

        # Total number of samples available
        self.num_nuscenes_total = len(self.all_nuscenes_indices)
        self.num_objaverse_total = len(self.objaverse_indices)


    def _linear_ratio(self):
        """Calculates the target real-world ratio for the current epoch with a linear ramp to max_lidar_ratio."""
        if self.num_nuscenes_total == 0:
            return 0.0
        if self.num_objaverse_total == 0:
            return 1.0

        # If before warmup ends, ratio is 0
        if self.epoch < self.warmup_epochs:
            return 0.0

        # If warmup is disabled or total epochs is invalid relative to warmup
        if self.total_epochs <= self.warmup_epochs:
            return self.max_lidar_ratio if self.num_objaverse_total > 0 else 1.0

        ramp_duration = self.total_epochs - self.warmup_epochs
        ramp_progress = self.epoch - self.warmup_epochs

        # Calculate the linear ratio: starts at 0.0 after warmup, increases to max_lidar_ratio by total_epochs
        # Ensure progress doesn't exceed duration (e.g., if epoch > total_epochs)
        fraction_complete = min(1.0, ramp_progress / ramp_duration)
        target_ratio = fraction_complete * self.max_lidar_ratio

        # Ensure the ratio does not exceed max_lidar_ratio
        return min(self.max_lidar_ratio, target_ratio)
    
    def _const_ratio(self):
        """Returns a constant target ratio of max_lidar_ratio."""
        if self.num_nuscenes_total == 0:
            return 0.0
        if self.num_objaverse_total == 0:
            return 1.0
        return self.max_lidar_ratio

    def __iter__(self):
        # Create a generator for this epoch, seeded for reproducibility across ranks
        # Including rank in the seed ensures each rank samples *independently* but deterministically
        g = np.random.Generator(np.random.PCG64(self.seed + self.epoch + self.rank))

        for i in range(self.num_batches):
            # Calculate target counts based on ratio for the current epoch
            nuscenes_ratio = self._linear_ratio() if not self.static else self._const_ratio()
            target_nuscenes_count = round(nuscenes_ratio * self.batch_size)

            # Ensure counts are within bounds and sum to batch_size
            target_nuscenes_count = max(0, min(target_nuscenes_count, self.batch_size))
            target_objaverse_count = self.batch_size - target_nuscenes_count

            # Determine actual number to sample, considering if datasets are empty
            actual_nuscenes_to_sample = target_nuscenes_count if self.num_nuscenes_total > 0 else 0
            actual_objaverse_to_sample = target_objaverse_count if self.num_objaverse_total > 0 else 0
            
            # for logging
            self.nuscenes_in_batch = actual_nuscenes_to_sample

            # If datasets are empty, counts might not sum to batch_size
            total_sampled = actual_nuscenes_to_sample + actual_objaverse_to_sample

            # Handle drop_last: if we couldn't fill the batch (due to empty dataset), skip
            if total_sampled < self.batch_size and self.drop_last:
                # This should only happen if one or both datasets are empty
                print(f"Rank {self.rank} Warning: Skipping batch {i} due to insufficient samples and drop_last=True.")
                continue # Skip this batch iteration

            # Sample indices using the seeded generator *with replacement*
            nuscenes_batch_indices = np.array([], dtype=np.int64)
            if actual_nuscenes_to_sample > 0:
                nuscenes_batch_indices = g.choice(
                    self.all_nuscenes_indices,
                    size=actual_nuscenes_to_sample,
                    replace=True # Sample with replacement
                )

            objaverse_batch_indices = np.array([], dtype=np.int64)
            if actual_objaverse_to_sample > 0:
                objaverse_batch_indices = g.choice(
                    self.objaverse_indices,
                    size=actual_objaverse_to_sample,
                    replace=True # Sample with replacement
                )

            # Combine indices
            # Handle case where one might be empty if total_sampled < batch_size and not drop_last
            if len(nuscenes_batch_indices) > 0 and len(objaverse_batch_indices) > 0:
                batch_indices_np = np.concatenate((nuscenes_batch_indices, objaverse_batch_indices))
            elif len(nuscenes_batch_indices) > 0:
                batch_indices_np = nuscenes_batch_indices
            elif len(objaverse_batch_indices) > 0:
                batch_indices_np = objaverse_batch_indices
            else:
                # Should not happen if drop_last=True and total_sampled < batch_size
                # If drop_last=False, yield empty or handle as needed. Let's yield empty for now.
                batch_indices_np = np.array([], dtype=np.int64)


            # Shuffle the combined batch if required
            if self.shuffle and len(batch_indices_np) > 0:
                g.shuffle(batch_indices_np)

            yield batch_indices_np.tolist()


    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. The epoch is used to calculate the
        NuScenes/Objaverse ratio and to seed the random number generator.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __len__(self):
        # Return the fixed number of batches
        # Note: If drop_last=True, the actual number yielded might be less
        # if datasets are empty. __len__ typically returns the intended number.
        return self.num_batches

class DistributedBalancedBatchSampler(data.Sampler):
    """
    Distributed-compatible batch sampler that aims for a 50/50 split between
    NuScenes and Objaverse samples per GPU batch. It ensures all NuScenes samples
    are iterated through across all GPUs per epoch, while randomly sampling
    Objaverse samples.

    Args:
        dataset (Objects_Joint): The dataset to sample from.
        batch_size (int): Per-GPU batch size.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, `world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within `num_replicas`.
            By default, `rank` is retrieved from the current distributed group.
        shuffle (bool, optional): If ``True`` (default), shuffle the NuScenes indices
            every epoch.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``. Default: ``False``.
            Note: In distributed mode, dropping the last batch applies per-rank
            based on the NuScenes data partition. If set to False, it might
 None, shuffle=True, seed=0, drop_last=False, total_epochs=None, warmup_epochs=None, total_batches=None):           result in slightly different batch sizes for the last batch across ranks.
    """
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if not isinstance(dataset, Objects_Joint):
            raise TypeError("Dataset must be an instance of Objects_Joint")
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size # Per-GPU batch size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0 # Track epoch for shuffling

        # Get all indices
        all_nuscenes_indices = np.array([i for i, data_info in enumerate(dataset.datapath) if data_info[0] == "nuscenes"], dtype=np.int64)
        self.objaverse_indices = np.array([i for i, data_info in enumerate(dataset.datapath) if data_info[0] == "objaverse"], dtype=np.int64)

        if len(all_nuscenes_indices) == 0:
            raise ValueError("Dataset must contain samples from NuScenes for balanced sampling.")
        if len(self.objaverse_indices) == 0:
            print(f"Rank {self.rank} Warning: Dataset does not contain Objaverse samples. Batches will only contain NuScenes.")
            self.nuscenes_batch_size = self.batch_size # Fill entire per-gpu batch with nuscenes
        else:
            self.nuscenes_batch_size = 0
            #if self.nuscenes_batch_size == 0 and self.batch_size > 0:
            #    self.nuscenes_batch_size = 1

        # Create a shuffled list of *all* nuscenes indices (consistent across ranks for epoch 0)
        g = np.random.Generator(np.random.PCG64(self.seed))
        indices_shuffled_globally = g.permutation(all_nuscenes_indices)

        # Assign indices to ranks
        # Example: world_size=4, rank=1 -> indices[1], indices[5], indices[9], ...
        self.nuscenes_indices_this_rank = indices_shuffled_globally[self.rank::self.num_replicas]
        self.num_nuscenes_this_rank = len(self.nuscenes_indices_this_rank)
        # ----------------------------------------------------

        # Calculate number of batches based on this rank's NuScenes data
        if self.nuscenes_batch_size > 0:
            if self.drop_last:
                # Number of batches based on full NuScenes batches for this rank
                self.num_batches_this_rank = self.num_nuscenes_this_rank // self.nuscenes_batch_size
            else:
                # Number of batches needed to cover all NuScenes for this rank
                self.num_batches_this_rank = math.ceil(self.num_nuscenes_this_rank / self.nuscenes_batch_size)
        else: # Should not happen if nuscenes_indices exist, but as fallback
             # If only objaverse, distribute based on objaverse indices (less ideal)
            num_objaverse_total = len(self.objaverse_indices)
            num_samples_per_rank = math.ceil(num_objaverse_total / self.num_replicas)
            self.num_batches_this_rank = math.ceil(num_samples_per_rank / self.batch_size) if not self.drop_last else num_samples_per_rank // self.batch_size

        # Ensure all ranks have the same number of batches using all_reduce (optional but good practice)
        # This requires initializing the distributed process group beforehand
        if dist.is_available() and dist.is_initialized():
            num_batches_tensor = torch.tensor(self.num_batches_this_rank, device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.MAX)
            self.num_batches = num_batches_tensor.item()
        else:
            # Fallback if distributed not initialized (e.g., single GPU run)
            self.num_batches = self.num_batches_this_rank


    def __iter__(self):
        # Shuffle NuScenes indices for this rank at the start of each epoch
        if self.shuffle:
            # Use epoch number in seed for shuffling consistency across ranks per epoch
            g = np.random.Generator(np.random.PCG64(self.seed + self.epoch))
            indices_this_rank_shuffled = g.permutation(self.nuscenes_indices_this_rank)
        else:
            indices_this_rank_shuffled = self.nuscenes_indices_this_rank

        nuscenes_ptr = 0
        len_nuscenes_this_rank = len(indices_this_rank_shuffled)

        # Use self.num_batches which is synchronized across ranks
        for i in range(self.num_batches):
            # Calculate start and end pointers for NuScenes batch for this rank
            start = nuscenes_ptr
            end = nuscenes_ptr + self.nuscenes_batch_size
            current_batch_nuscenes = np.array([], dtype=np.int64) # Default empty

            if start < len_nuscenes_this_rank:
                # Slice the shuffled NumPy array for this rank
                current_batch_nuscenes = indices_this_rank_shuffled[start:min(end, len_nuscenes_this_rank)]
                nuscenes_ptr = end
            # else: NuScenes for this rank exhausted, current_batch_nuscenes remains empty

            # Check for drop_last condition for this rank's NuScenes data
            # Note: If not drop_last, ranks might have slightly different last batch sizes
            # if their NuScenes partitions weren't perfectly divisible.
            is_last_batch = (i == self.num_batches - 1)
            num_fetched_nuscenes = len(current_batch_nuscenes)

            if self.drop_last and num_fetched_nuscenes < self.nuscenes_batch_size and num_fetched_nuscenes > 0 and not is_last_batch:
                # This condition is tricky in distributed setting. If we drop here,
                # other ranks might continue, leading to desync.
                # A common strategy is to pad or let the last batch be smaller.
                # For simplicity with drop_last=True, we might just yield fewer batches
                # if the initial calculation `self.num_batches_this_rank` used floor division.
                # Since we used floor division for drop_last=True earlier, this break isn't strictly needed.
                # Let's rely on self.num_batches being calculated correctly with floor division for drop_last=True.
                pass # Relying on num_batches calculation for drop_last  
            
            # Determine how many Objaverse samples are needed for this GPU's batch
            num_objaverse_needed = self.batch_size - len(current_batch_nuscenes)
            current_batch_objaverse = np.array([], dtype=np.int64) # Empty numpy array

            if num_objaverse_needed > 0 and len(self.objaverse_indices) > 0:
                # Sample Objaverse indices using NumPy (with replacement) - each rank samples independently
                current_batch_objaverse = np.random.choice(
                    self.objaverse_indices,
                    size=num_objaverse_needed,
                    replace=False
                )

            # Combine using NumPy concatenate
            if len(current_batch_nuscenes) > 0 or len(current_batch_objaverse) > 0:
                batch_indices_np = np.concatenate((current_batch_nuscenes, current_batch_objaverse))
                np.random.shuffle(batch_indices_np) # happens per-rank
                yield batch_indices_np.tolist()
            elif not self.drop_last and is_last_batch:
                # If not dropping last and this rank has no more data, yield empty list?
                # Or better: ensure num_batches calculation handles padding implicitly.
                # With ceil division for drop_last=False, this rank might just finish early.
                # The DataLoader should handle the StopIteration. Let's not yield empty.
                pass


    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __len__(self):
        # Return the synchronized number of batches
        return self.num_batches
    
@DATASETS.register_module()
class Objects_Joint(data.Dataset):
    def __init__(self, config):
        """
        Joint dataloader for NuScenes objects and Objaverse objects
        Args:
            config: A config object with dataset configuration
        """
        self.subset = getattr(config, 'subset', 'train')  # 'train' or 'test'   
        if self.subset != 'train':
            raise ValueError("Objects_Joint dataset is only available for training. Validation/test not supported.")
        
        self.augment = self.subset == 'train' # For augmentation in training
        self.excluded_nuscenes_classes = config.excluded_classes
        print(f"Excluding the following NuScenes classes from training: {self.excluded_nuscenes_classes}")
        
        self.no_motion_variant = getattr(config, 'NO_MOTION', False) # motion compensation for nuscenes
            
        #self.use_normals = getattr(config, 'USE_NORMALS', False)
        self.simulate_occlusion = getattr(config, 'sim_occlusion', False)
        if self.simulate_occlusion:
            print("Simulating occlusion for training.")
        
        self.nuscenes_median = 574
        self.use_height = getattr(config, 'USE_HEIGHT', False)
        
        self.uniform = getattr(config, 'uniform', True)
        self.process_data = getattr(config, 'process_data', True)
        self.cap_to_npoints = config.cap_to_npoints
        self.npoints = config.npoints

        # For ULIP training
        self.tokenizer = getattr(config, 'tokenizer', None)
        self.train_transform = getattr(config, 'train_transform', None)
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for Objects_Joint dataset.")
        
        # NuScenes configuration
        self.nuscenes_root = config.NUSCENES_PATH
        self.nuscenes_subset_dir = "train" if self.subset == 'train' else "val"
        
        # Objaverse configuration
        self.objaverse_root = config.OBJAVERSE_PATH
        self.objaverse_pc_dir = os.path.join(self.objaverse_root, "objaverse_pc_parallel")
        self.objaverse_img_dir = os.path.join(self.objaverse_root, "rendered_images_split_100")
        self.objaverse_caption_file = os.path.join(self.objaverse_root, "merged_data.json")
        self.use_colored_pc = getattr(config, 'use_colored_pc', True) # Use colored point clouds if available
        
        if self.use_colored_pc:
            print("Using colored point clouds from Objects-Joint.")
        else:
            print("Using XYZ point clouds from Objects-Joint.")
        
        # Load NuScenes data
        self._load_nuscenes_data()
        
        # Load Objaverse data
        self._load_objaverse_data()
        
        # Combine data paths
        self.datapath = self.nuscenes_datapath + self.objaverse_datapath
        print(f"The size of {self.subset} data is {len(self.datapath)} (NuScenes: {len(self.nuscenes_datapath)}, Objaverse: {len(self.objaverse_datapath)})")
    
    def _load_nuscenes_data(self):
        """Load NuScenes objects data"""
        
        # Load captions for training
        if self.subset == 'train':
            self.nuscenes_captions = {}
            caption_file = os.path.join(self.nuscenes_root, self.nuscenes_subset_dir, "captions.json")
            try:
                with open(caption_file, 'r') as f:
                    self.nuscenes_captions = json.load(f)
            except Exception as e:
                raise ValueError(f"Error loading NuScenes captions from {caption_file}: {e}")
        
        # Load the per-object class labels from labels.txt
        label_file = os.path.join(self.nuscenes_root, self.nuscenes_subset_dir, "labels.txt")
        try:
            with open(label_file, 'r') as f:
                labels_dict = {
                    line.strip().split(':')[0]: line.strip().split(':')[1] 
                    for line in f
                }
        except Exception as e:
            raise ValueError(f"Error loading labels from {label_file}: {e}")
        
        # Path to point clouds and images
        self.nuscenes_pc_dir = os.path.join(self.nuscenes_root, self.nuscenes_subset_dir, "pc") if not self.no_motion_variant else os.path.join(self.nuscenes_root, self.nuscenes_subset_dir, "pc_nomotion")
        self.nuscenes_img_dir = os.path.join(self.nuscenes_root, self.nuscenes_subset_dir, "images")
        
        # Get all point cloud files
        pc_files = [f for f in os.listdir(self.nuscenes_pc_dir) if f.endswith(".hdf5")]
        
        self.nuscenes_datapath = []
        for pc_file in tqdm(pc_files, desc=f"Loading NuScenes {self.subset} data"):
            instance_id = pc_file[:-5]  # remove the trailing ".hdf5"
            pc_path = os.path.join(self.nuscenes_pc_dir, pc_file)
            img_path = os.path.join(self.nuscenes_img_dir, pc_file)  # Same name, different directory
            
            # Skip excluded classes
            if labels_dict.get(instance_id) in self.excluded_nuscenes_classes:
                continue
            
            # Check if image file exists
            if not os.path.exists(img_path):
                print(f"Warning: No image file found for NuScenes instance {instance_id}, skipping")
                continue
            
            # Open the point cloud file and get all keys
            try:
                with h5py.File(pc_path, 'r') as f:
                    pc_keys = list(f.keys())
                    
                    if not pc_keys:
                        raise ValueError(f"No point cloud keys found in {pc_path}")
                    
                    # Add each key as a separate data point
                    for key in pc_keys:
                        self.nuscenes_datapath.append(("nuscenes", instance_id, pc_path, img_path, key))
            except Exception as e:
                raise ValueError(f"Error loading NuScenes point cloud file {pc_path}: {str(e)}")
        
    
    def _load_objaverse_data(self):
        """Load Objaverse objects data without preselecting image keys"""
        
        try:   
            # Load captions from merged_data.json
            with open(self.objaverse_caption_file, 'r') as f:
                self.objaverse_captions = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading Objaverse captions from {self.objaverse_caption_file}: {e}")
            
        # Get all point cloud files
        pc_files = [f for f in os.listdir(self.objaverse_pc_dir) if f.endswith(".npz")]
        
        self.objaverse_datapath = []
        for pc_file in tqdm(pc_files, desc=f"Loading Objaverse {self.subset} data"):
            instance_id = pc_file.split('_')[0]
            pc_path = os.path.join(self.objaverse_pc_dir, pc_file)
            img_path = os.path.join(self.objaverse_img_dir, f"{instance_id}.hdf5")
            
            # Check if image file exists
            if not os.path.exists(img_path):
                print(f"Warning: No image file found for Objaverse instance {instance_id}, skipping")
                continue
            
            self.objaverse_datapath.append(("objaverse", instance_id, pc_path, img_path, None))

            """
            # Check if the HDF5 file has valid images with captions
            try:
                with h5py.File(img_path, 'r') as f:
                    img_keys = list(f.keys())
                    
                    # Skip if no images available
                    if not img_keys:
                        continue
                    
                    # Check if at least one image has a caption
                    #has_valid_caption = False
                    #for img_key in img_keys:
                    #    caption_key = f"/export/einstein-vision/3d_vision/objaverse/render_images_split_100/{instance_id}/{img_key}.png"
                    #    if caption_key in self.objaverse_captions:
                    #        has_valid_caption = True
                    #        break
                    
                    #if not has_valid_caption:
                    #    continue
                    
                    self.objaverse_datapath.append(("objaverse", instance_id, pc_path, img_path, None, -1, "objaverse"))
            except Exception as e:
                print(f"Error loading Objaverse HDF5 file {img_path}: {str(e)}")
                continue
            """
    
    def _load_and_preprocess_nuscenes_points(self, file_path, key):
        """Loads points from the NuScenes HDF5 file for a specific key"""
        try:
            with h5py.File(file_path, 'r') as f:
                point_set = f[key][:]
        except Exception as e:
            raise ValueError(f"Error loading NuScenes point cloud file {file_path}: {str(e)}")
        
        # Handle different point cloud formats
        if point_set.shape[1] < 3:
            point_set = point_set.transpose(0, 2, 1)
        
        # Normalize the point cloud
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            
        # Cap to npoints if specified
        if self.cap_to_npoints and point_set.shape[0] > self.npoints:
            point_set = farthest_point_sample(point_set, self.npoints) # FPS as it can be sparse
        """
        # Add height dimension if needed
        if self.use_height:
            gravity_dim = 1  # Y-axis is usually up in nuScenes
            height_array = point_set[:, gravity_dim:gravity_dim + 1] - point_set[:, gravity_dim].min()
            point_set = np.concatenate((point_set, height_array), axis=1)
        """             
        return point_set
    
    def _load_and_preprocess_objaverse_points(self, file_path):
        """Loads points from the Objaverse NPZ file"""
        try:
            with np.load(file_path) as data:
                point_set = data['point_cloud']
        except Exception as e:
            raise ValueError(f"Error loading Objaverse point cloud file {file_path}: {str(e)}")
       
        # Normalize the point cloud coordinates
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        # simulate occlusion
        if self.simulate_occlusion:
            tries = 0
            INV_RAD_PARAM = 2
            
            occluded, _ = simulate_occlusion(point_set, param=INV_RAD_PARAM)
            while occluded.shape[0] < 500 and tries < 5:
                occluded, _ = simulate_occlusion(point_set, param=INV_RAD_PARAM)
                tries += 1
            if occluded.shape[0] >= 500:
                point_set = occluded
            
        if self.cap_to_npoints and point_set.shape[0] > self.npoints:
            indices = np.random.choice(point_set.shape[0], self.npoints, replace=False)
            point_set = point_set[indices, :]
        
        # Randomly subsample to get nuscenes_median points
        #if point_set.shape[0] > self.nuscenes_median:
        #    indices = np.random.choice(point_set.shape[0], self.nuscenes_median, replace=False)
        #    point_set = point_set[indices, :]
            
        # Visualize the point cloud using Open3D (debugging)
        #import open3d as o3d
        #o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_set[:, :3]))])
        """ 
        # Add height dimension if needed
        if self.use_height:
            gravity_dim = 1  # Assuming Y-axis is up
            height_array = point_set[:, gravity_dim:gravity_dim + 1] - point_set[:, gravity_dim].min()
            point_set = np.concatenate((point_set, height_array), axis=1)
        """ 
        return point_set
    
    def _load_image(self, img_path, dataset_type):
        """Loads a random image from the HDF5 file"""
        try:
            with h5py.File(img_path, 'r') as f:
                img_keys = list(f.keys())
                
                if not img_keys:
                    raise ValueError(f"No images found in {img_path}")
                
                img_key = random.choice(img_keys)
                img_data = f[img_key][:]
                
                # Convert to PIL Image
                if len(img_data.shape) == 3:
                # Determine if CHW or HWC format based on shape dimensions
                    if img_data.shape[0] <= 4 and img_data.shape[0] < img_data.shape[1]:
                        # CHW format (channels first)
                        img_data = np.transpose(img_data, (1, 2, 0))
                    
                    # Now all images should be HWC format
                    # Handle different channel configurations
                    if img_data.shape[2] == 3:  # RGB
                        img = Image.fromarray(img_data)
                    elif img_data.shape[2] == 4:  # RGBA
                        img = Image.fromarray(img_data).convert('RGB')
                    elif img_data.shape[2] == 1:  # Grayscale
                        img = Image.fromarray(img_data.squeeze()).convert('RGB')
                    else:
                        raise ValueError(f"Unsupported number of channels: {img_data.shape[2]}")
                else:
                    raise ValueError(f"Unexpected image dimensions: {img_data.shape}")
                
                return img, img_key
        except Exception as e:
            raise ValueError(f"Error loading image from file {img_path} with key {img_key}: {str(e)}")
            #blank = Image.new('RGB', (224, 224), (100, 100, 100))
            #return blank, "fallback"
    
    def _get_caption(self, instance_id, img_key, dataset_type):
        """Gets caption for the given instance ID and image key"""
        if dataset_type == "nuscenes":
            if instance_id in self.nuscenes_captions:
                if img_key in self.nuscenes_captions[instance_id]:
                    return self.nuscenes_captions[instance_id][img_key]
                else:
                    raise ValueError(f"No captions found for NuScenes instance {instance_id}/{img_key}.")
            else:
                raise ValueError(f"No captions found for NuScenes instance {instance_id}.")
            
        elif dataset_type == "objaverse":  # objaverse
            caption_key = f"/export/einstein-vision/3d_vision/objaverse/render_images_split_100/{instance_id}/{img_key}"
            if caption_key in self.objaverse_captions:
                return random.choice(self.objaverse_captions[caption_key])
            else:
                raise ValueError(f"No captions found for Objaverse instance {instance_id}/{img_key}.")
                #return "A 3D object"
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def __getitem__(self, index):
        """Returns data in the expected format based on training or testing"""
        dataset_type, instance_id, data_path, img_path, key = self.datapath[index]
        
        # Load point cloud based on dataset type
        if dataset_type == "nuscenes":
            point_set = self._load_and_preprocess_nuscenes_points(data_path, key)
        elif dataset_type == "objaverse":
            point_set = self._load_and_preprocess_objaverse_points(data_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Apply augmentation if in training mode
        if self.augment:
            points_np = point_set.copy()
            points_np = random_scale_point_cloud(points_np[None, ...])
            points_np = shift_point_cloud(points_np)
            points_np = rotate_perturbation_point_cloud(points_np)
            points_np = rotate_point_cloud(points_np)
            points_np = random_point_dropout(points_np)
            points_np = points_np.squeeze()
            #points_np = points_np[~np.isinf(points_np).any(axis=1)]
        else:
            points_np = point_set.copy()
        
        if not self.use_colored_pc:
            points_np = points_np[:, :3]  # Keep only XYZ if not using colored point clouds
            
        current_points = torch.from_numpy(points_np).float()
        
        """
        # Optional: Ensure fixed number of points (pad or subsample)
        num_current_points = points_np.shape[0]
        num_features = points_np.shape[1] # Get number of features (e.g., 3 for XYZ, 6 for XYZRGB)

        if num_current_points > self.npoints:
            # Randomly subsample
            indices = np.random.choice(num_current_points, self.npoints, replace=False)
            points_np = points_np[indices, :]
        elif num_current_points < self.npoints:
            # Pad with np.inf
            padding_size = self.npoints - num_current_points
            padding = np.full((padding_size, num_features), np.inf, dtype=points_np.dtype)
            points_np = np.concatenate([points_np, padding], axis=0)
        """
        
        # If in training mode, return in format compatible with main.py
        if self.tokenizer is not None:
            # Load the image
            image_pil, img_key = self._load_image(img_path, dataset_type)
            
            # Apply transformation
            if self.train_transform is not None:
                image = self.train_transform(image_pil)
            else:
                # Convert to tensor if no transform provided
                image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
            
            # Get caption
            caption = self._get_caption(instance_id, img_key, dataset_type)
            
            # Tokenize caption
            tokenized_captions = self.tokenizer(caption)
            
            # For training
            unique_key = f"{dataset_type}+{instance_id}+{key}+{img_key}"
            return unique_key, tokenized_captions, current_points, image
    
    def __len__(self):
        return len(self.datapath)

@DATASETS.register_module()
class Objaverse(data.Dataset):
    def __init__(self, config):
        """
        Dataloader for Objaverse objects only.
        Args:
            config: A config object with dataset configuration
        """
        self.subset = getattr(config, 'subset', 'train')  # 'train' or 'test'
        if self.subset != 'train':
            raise ValueError("Objaverse dataset is only available for training. Validation/test not supported.")
        
        self.augment = self.subset == 'train' # For augmentation in training
            
        self.simulate_occlusion = getattr(config, 'sim_occlusion', False)
        if self.simulate_occlusion:
            print("Simulating occlusion for training.")

        self.npoints = config.npoints # npoints might be needed for padding/sampling if not handled elsewhere
        self.nuscenes_median = 574 # Keep for consistency if Objaverse points are sampled to this number
        self.cap_to_npoints = config.cap_to_npoints
        self.use_height = getattr(config, 'use_height', False) # Changed from USE_HEIGHT

        # For ULIP training
        self.tokenizer = getattr(config, 'tokenizer', None)
        self.train_transform = getattr(config, 'train_transform', None)
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for Objaverse dataset.")

        # Objaverse configuration
        self.objaverse_root = config.DATA_PATH # Assuming DATA_PATH points to Objaverse root for this config
        self.objaverse_pc_dir = os.path.join(self.objaverse_root, "objaverse_pc_parallel")
        self.objaverse_img_dir = os.path.join(self.objaverse_root, "rendered_images_split_100")
        self.objaverse_caption_file = os.path.join(self.objaverse_root, "merged_data.json")
        self.use_colored_pc = getattr(config, 'use_colored_pc', True) # Use colored point clouds if available
        
        if self.use_colored_pc:
            print("Using colored point clouds from Objaverse.")
        else:
            print("Using XYZ point clouds from Objaverse.")

        # Load Objaverse data
        self._load_objaverse_data()

        print(f"The size of {self.subset} Objaverse data is {len(self.datapath)}")

    def _load_objaverse_data(self):
        """Load Objaverse objects data without preselecting image keys"""

        # Load captions from merged_data.json
        try:
            with open(self.objaverse_caption_file, 'r') as f:
                self.objaverse_captions = json.load(f)
        except Exception as e:
             raise ValueError(f"Error loading Objaverse captions from {self.objaverse_caption_file}: {e}")

        # Get all point cloud files
        try:
            pc_files = [f for f in os.listdir(self.objaverse_pc_dir) if f.endswith(".npz")]
        except FileNotFoundError:
             raise FileNotFoundError(f"Objaverse point cloud directory not found: {self.objaverse_pc_dir}")

        self.datapath = []
        for pc_file in tqdm(pc_files, desc=f"Loading Objaverse {self.subset} data"):
            # Extract instance_id, assuming format like 'instanceid_something.npz'
            instance_id = pc_file.split('_')[0]
            pc_path = os.path.join(self.objaverse_pc_dir, pc_file)
            # Image path uses HDF5 format based on Objects_Joint
            img_path = os.path.join(self.objaverse_img_dir, f"{instance_id}.hdf5")

            # Check if image file exists (only strictly needed for training)
            if not os.path.exists(img_path):
                if self.subset == 'train':
                    print(f"Warning: No image file found for Objaverse instance {instance_id}, skipping")
                    continue
            
            """
            # check if files are corrupted
            # PC
            try:
                _ = np.load(pc_path)
            except Exception as e:
                print(f"Corrupted Objaverse point cloud file {pc_path}: {str(e)}, skipping")
                continue
            # IMG
            try:
                _ = h5py.File(img_path, 'r')
            except Exception as e:
                print(f"Corrupted Objaverse image file {img_path}: {str(e)}, skipping")
                continue
            """
            # Check if caption exists for this instance (implicitly checked in _get_caption)
            # We add the item here and handle potential missing captions in __getitem__

            self.datapath.append((instance_id, pc_path, img_path))


    def _load_and_preprocess_points(self, file_path):
        """Loads points from the Objaverse NPZ file"""
        try:
            with np.load(file_path) as data:
                point_set = data['point_cloud']
        except Exception as e:
            raise ValueError(f"Error loading Objaverse point cloud file {file_path}: {str(e)}")

        # Normalize the point cloud coordinates
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
         # simulate occlusion
        if self.simulate_occlusion:
            tries = 0
            INV_RAD_PARAM = 2
            
            occluded, _ = simulate_occlusion(point_set, param=INV_RAD_PARAM)
            while occluded.shape[0] < 500 and tries < 5:
                occluded, _ = simulate_occlusion(point_set, param=INV_RAD_PARAM)
                tries += 1
            if occluded.shape[0] >= 500:
                point_set = occluded

        # Randomly subsample to get nuscenes_median points
        #if point_set.shape[0] > self.nuscenes_median:
        #    indices = np.random.choice(point_set.shape[0], self.nuscenes_median, replace=False)
        #    point_set = point_set[indices, :]
        
        if self.cap_to_npoints and point_set.shape[0] > self.npoints:
            indices = np.random.choice(point_set.shape[0], self.npoints, replace=False)
            point_set = point_set[indices, :]

        # Add height dimension if needed
        #if self.use_height:
        #    gravity_dim = 1  # Assuming Y-axis is up
        #    height_array = point_set[:, gravity_dim:gravity_dim + 1] - point_set[:, gravity_dim].min()
        #    point_set = np.concatenate((point_set, height_array), axis=1)

        return point_set

    def _load_image(self, img_path, instance_id):
        """Loads a random image from the HDF5 file for Objaverse"""
        if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found for training instance {instance_id}: {img_path}")

        try:
            with h5py.File(img_path, 'r') as f:
                img_keys = list(f.keys())

                if not img_keys:
                    raise ValueError(f"No images found in {img_path}")

                img_key = random.choice(img_keys)
                img_data = f[img_key][:]

                # Convert to PIL Image (copied from Objects_Joint)
                if len(img_data.shape) == 3:
                    if img_data.shape[0] <= 4 and img_data.shape[0] < img_data.shape[1]:
                        img_data = np.transpose(img_data, (1, 2, 0)) # CHW to HWC

                    if img_data.shape[2] == 3: img = Image.fromarray(img_data)
                    elif img_data.shape[2] == 4: img = Image.fromarray(img_data).convert('RGB')
                    elif img_data.shape[2] == 1: img = Image.fromarray(img_data.squeeze()).convert('RGB')
                    else: raise ValueError(f"Unsupported image channels: {img_data.shape[2]}")
                else:
                    raise ValueError(f"Unexpected image dimensions: {img_data.shape}")

                return img, img_key
        except Exception as e:
            raise ValueError(f"Error loading image from file {img_path} with key {img_key}: {str(e)}")
            #blank = Image.new('RGB', (224, 224), (100, 100, 100))
            #return blank, "fallback_load_error"

    def _get_caption(self, instance_id, img_key):
        """Gets caption for the given Objaverse instance ID and image key"""

        caption_key = f"/export/einstein-vision/3d_vision/objaverse/render_images_split_100/{instance_id}/{img_key}" 
        if caption_key in self.objaverse_captions:
            """
            # return all captions from self.objaverse_captions[caption_key]
            captions = []
            for caption in self.objaverse_captions[caption_key]:
                captions.append(caption)
            return captions
            """
            #return random.choice(self.objaverse_captions[caption_key])
            return self.objaverse_captions[caption_key][0]
        else:
            raise ValueError(f"No captions found for Objaverse instance {instance_id}/{img_key}.")

    def __getitem__(self, index):
        """Returns data in the expected format based on training or testing"""
        instance_id, pc_path, img_path = self.datapath[index]

        # Load point cloud
        point_set = self._load_and_preprocess_points(pc_path)

        # Apply augmentation if in training mode
        if self.augment:
            points_np = point_set.copy()
            points_np = random_scale_point_cloud(points_np[None, ...])
            points_np = shift_point_cloud(points_np)
            points_np = rotate_perturbation_point_cloud(points_np)
            points_np = rotate_point_cloud(points_np)
            points_np = random_point_dropout(points_np)
            #points_np = points_np[~np.isinf(points_np).any(axis=1)] # if the removed point becomes inf 
            points_np = points_np.squeeze()
        else:
            points_np = point_set.copy()

        if not self.use_colored_pc:
            points_np = points_np[:, :3]  # Keep only XYZ if not using colored point clouds
            
        current_points = torch.from_numpy(points_np).float()

        # Return format depends on subset (train vs val/test)
        if self.tokenizer is not None:
            # Load the image
            image_pil, img_key = self._load_image(img_path, instance_id)
            """
            #display image using matplotlib
            plt.imshow(image_pil)
            plt.axis('off')
            plt.show()
            
            # visualize PC with open3d
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(current_points['coord'])
            pcd.colors = o3d.utility.Vector3dVector(current_points['color'])
            o3d.visualization.draw_geometries([pcd])
            """
            
            captions = self._get_caption(instance_id, img_key)
            #print(f"Captions: {captions}")
            
            # Apply image transformation
            if self.train_transform is not None:
                image = self.train_transform(image_pil)
                #print("Type of img after transform: ", type(image))
            #else:
                # Default transform if none provided
            #    image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0

            # Tokenize captions
            tokenized_captions = self.tokenizer(captions)

            # Unique key for training
            unique_key = f"{instance_id}+{img_key}"
            return unique_key, tokenized_captions, current_points, image

    def __len__(self):
        return len(self.datapath)
    
@DATASETS.register_module()
class NuScenes_Objects(data.Dataset):
    def __init__(self, config):
        """
        Args:
            config: A config object/namespace with dataset configuration
        """
        self.subset = getattr(config, 'subset', 'train')  # 'train' or 'test'
        self.augment = self.subset == 'train' # For augmentation in training
        self.class_labels = getattr(config, 'validate_dataset_name', 'nuscenes_objects') # For validation dataset name
        self.no_motion_variant = getattr(config, 'NO_MOTION', False)
        
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.cap_to_npoints = config.cap_to_npoints
        self.skip_labels = config.excluded_classes
        self.skip_labels = self.skip_labels + ['void'] if self.subset == 'test' else self.skip_labels # only skip void if testing
        print(f"Excluding classes: {self.skip_labels}")
        
        #self.use_normals = getattr(config, 'USE_NORMALS', False)
        self.use_height = getattr(config, 'USE_HEIGHT', False)
        self.subset_dir = "train" if self.subset == 'train' else "val"
        self.generate_from_raw_data = getattr(config, 'generate_from_raw_data', False)
        self.process_data = getattr(config, 'process_data', True)
        self.uniform = getattr(config, 'uniform', True)
        self.use_caption_templates = False
        self.use_colored_pc = getattr(config, 'use_colored_pc', True) # Use colored point clouds if available
        self.dataset_name = getattr(config, 'dataset_name', 'nuscenes_objects')  # Default to 'nuscenes_objects'
        self.force_test_return = getattr(config, 'FORCE_TEST_RETURN', False)  # Force test return format even in training
        if self.force_test_return:
            print("FORCE_TEST_RETURN is enabled: Dataset will return test format even in training mode.")
        
        
        if self.use_colored_pc:
            print(f"Using colored point clouds from {self.dataset_name}.")
        else:
            print(f"Using XYZ point clouds from {self.dataset_name}.")
        
        # For ULIP training - store required components
        self.tokenizer = getattr(config, 'tokenizer', None)
        self.train_transform = getattr(config, 'train_transform', None)
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for NuScenes_Objects dataset.")

        """
        # Load templates for text prompts
        self.templates = []
        if hasattr(config, 'pretrain_dataset_prompt') or hasattr(config, 'validate_dataset_prompt'):
            prompt_key = getattr(config, 'pretrain_dataset_prompt' if self.subset == 'train' else 'validate_dataset_prompt', None)
            if prompt_key:
                try:
                    with open('./data/templates.json') as f:
                        templates_data = json.load(f)
                        if prompt_key in templates_data:
                            self.templates = templates_data[prompt_key]
                except Exception as e:
                    raise ValueError(f"Error loading prompt templates: {e}")
        """
        
        # Load the per-object class labels from labels.txt
        label_file = os.path.join(self.root, self.subset_dir, "labels.txt")
        try:
            with open(label_file, 'r') as f:
                self.labels_dict = {
                    line.strip().split(':')[0]: line.strip().split(':')[1] 
                    for line in f
                }
        except Exception as e:
            raise ValueError(f"Error loading labels from {label_file}: {e}")
            #self.labels_dict = json.load(f)
            
        # Load captions for training
        if self.subset == 'train':
            self.captions = {}
            caption_file = os.path.join(self.root, self.subset_dir, "captions.json")
            if os.path.exists(caption_file):
                try:
                    with open(caption_file, 'r') as f:
                        self.captions = json.load(f)
                except Exception as e:
                    raise ValueError(f"Error loading captions from {caption_file}: {e}")
            else:
                print(f"Warning: No captions found at {caption_file}, using class names instead.")
                self.use_caption_templates = True  # Fallback to class names if no captions are found
        
        if self.subset == 'test':
            # Collect the distinct classes and assign each a numeric ID
            ordered_labels_path = os.path.join("./data", 'labels.json')
            try:
                with open(ordered_labels_path, 'r') as f:
                    #classes = json.load(f)["nuscenes_objects_official"] # nuscenes_objects or truckscenes
                    #classes = json.load(f)["truckscenes_objects_official"] # truckscenes or nuscenes
                    classes = json.load(f)[self.dataset_name]  # Use official NuScenes-Objects labels
            except Exception as e:
                raise ValueError(f"Error loading ordered label list from {ordered_labels_path}: {e}")

            self.classes = {name: idx for idx, name in enumerate(classes)}
            self.shape_names = classes

        # Create datapath with all point clouds from all files/keys
        self.pc_dir = os.path.join(self.root, self.subset_dir, "pc") if not self.no_motion_variant else os.path.join(self.root, self.subset_dir, "pc_nomotion")
        self.img_dir = os.path.join(self.root, self.subset_dir, "images")
        
        # Get all point cloud files
        pc_files = [f for f in os.listdir(self.pc_dir) if f.endswith(".hdf5")]
        pc_files.sort()
        
        self.datapath = []
        for pc_file in tqdm(pc_files, desc=f"Loading {self.dataset_name} {self.subset} data"):
            instance_id = pc_file[:-5]  # remove the trailing ".hdf5"
            pc_path = os.path.join(self.pc_dir, pc_file)
            img_path = os.path.join(self.img_dir, pc_file)  # Same name, different directory
            
            # Check if image file exists
            if not os.path.exists(img_path) and self.subset == 'train':
                print(f"Warning: No image file found for {instance_id}, skipping")
                continue
                
            # Get label
            if (instance_id not in self.labels_dict):
                print(f"Warning: No label found for {instance_id}, skipping")
                continue
                
            # skip special labels like 'void'
            label_name = self.labels_dict[instance_id]
            if label_name in self.skip_labels:
                continue
            
            label_idx = self.classes[label_name] if self.subset == 'test' else -1
            
            # Open the point cloud file and get all keys
            try:
                with h5py.File(pc_path, 'r') as f:
                    pc_keys = list(f.keys())
                    
                    if not pc_keys:
                        raise ValueError(f"No point cloud keys found in {pc_path}")
                    
                    # Add each key as a separate data point
                    for key in pc_keys:
                        self.datapath.append((instance_id, pc_path, img_path, key, label_idx))
            except Exception as e:
                raise ValueError(f"Error loading point cloud file {pc_path}: {str(e)}")

        print(f"The size of {self.subset} data is {len(self.datapath)}")

        # Cache for preloaded data if needed
        #self.list_of_points = None
        #self.list_of_labels = None

    def _load_and_preprocess_points(self, file_path, key):
        """Loads points from the HDF5 file for a specific key"""
        try:
            with h5py.File(file_path, 'r') as f:
                point_set = f[key][:]
        except Exception as e:
            raise ValueError(f"Error loading point cloud file {file_path}: {str(e)}")
        
        # Handle different point cloud formats
        if point_set.shape[1] < 3:
            # Some datasets might have shape [N, C, 3] instead of [N, 3, C]
            point_set = point_set.transpose(0, 2, 1)
        
        # Normalize the point cloud
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # Add height dimension if needed
        #if self.use_height:
        #    gravity_dim = 1  # Y-axis is usually up in nuScenes
        #    height_array = point_set[:, gravity_dim:gravity_dim + 1] - point_set[:, gravity_dim].min()
        #    point_set = np.concatenate((point_set, height_array), axis=1)

        # Ensure we have the right number of points
        if self.cap_to_npoints and point_set.shape[0] > self.npoints:
            object_id = f"{file_path}_{key}"
            object_seed = abs(hash(object_id)) % (2**32) if self.subset == 'test' else None
            
            point_set = farthest_point_sample(point_set, self.npoints, seed=object_seed)

        return point_set

    def _load_random_image(self, img_path, sid):
        """Loads a random image from the HDF5 file"""
        try:
            with h5py.File(img_path, 'r') as f:
                img_keys = list(f.keys())
                if not img_keys:
                    raise ValueError(f"No images found in {img_path}")
                
                # Randomly select an image
                random_key = random.choice(img_keys)
                img_data = f[random_key][:]
                
                # Convert to PIL Image
                if len(img_data.shape) == 3 and img_data.shape[2] == 3:  # HWC format
                    img = Image.fromarray(img_data)
                elif len(img_data.shape) == 3 and img_data.shape[0] == 3:  # CHW format
                    img = Image.fromarray(np.transpose(img_data, (1, 2, 0)))
                else:
                    raise ValueError(f"Unexpected image shape: {img_data.shape}")
                
                return img, random_key
        except Exception as e:
            raise ValueError(f"Error loading image for {sid}: {str(e)}")
            #blank = Image.new('RGB', (224, 224), (100, 100, 100))
            #return blank, "fallback"
    
    def _get_caption(self, instance_id, img_key):
        """Gets caption for the given instance ID and image key"""
        
        if self.use_caption_templates:
            # Use class label as caption if templates are used
            caption = self.labels_dict[instance_id]
            return caption
        
        if instance_id in self.captions.keys():
            caption = self.captions[instance_id][img_key]
        else: 
            raise ValueError(f"No caption found for instance {instance_id}.")
            #caption = self.labels_dict[instance_id]      
            
        return caption

    def __getitem__(self, index):
        """Returns data in the expected format based on training or testing"""
        instance_id, pc_path, img_path, pc_key, label_idx = self.datapath[index]
        
        # Load point cloud
        point_set = self._load_and_preprocess_points(pc_path, pc_key)
                
         # Apply augmentation if in training mode
        if self.augment:
            points_np = point_set.copy()
            points_np = random_scale_point_cloud(points_np[None, ...])
            points_np = shift_point_cloud(points_np)
            points_np = rotate_perturbation_point_cloud(points_np)
            points_np = rotate_point_cloud(points_np)
            points_np = random_point_dropout(points_np)
            points_np = points_np.squeeze()
            #points_np = points_np[~np.isinf(points_np).any(axis=1)]
        else:
            points_np = point_set.copy()
    
        if not self.use_colored_pc:
            points_np = points_np[:, :3]  # Keep only XYZ if not using colored point clouds
                
        current_points = torch.from_numpy(points_np).float()

        # If in training mode, return in format compatible with main.py
        if self.subset == 'train' and not self.force_test_return and self.tokenizer is not None:
            # Load a random image
            image_pil, img_key = self._load_random_image(img_path, instance_id)
            
            # Apply transformation
            if self.train_transform is not None:
                image = self.train_transform(image_pil)
            else:
                # Convert to tensor if no transform provided
                image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
            
            # Get caption - try to get specific one for this image key first
            caption = self._get_caption(instance_id, img_key)
            
            # Tokenize caption
            tokenized_captions = self.tokenizer(caption)
            
            # For training
            unique_key = f"{instance_id}+{pc_key}+{img_key}"
            return unique_key, tokenized_captions, current_points, image
        
        # For validation/testing
        # for Point-BERT tests
        """
        NPOINTS = 10000
        if points_np.shape[0] < NPOINTS:
            pad = np.full((NPOINTS - points_np.shape[0], points_np.shape[1]), np.inf)
            points_np = np.concatenate([points_np, pad], axis=0)
        elif points_np.shape[0] > NPOINTS:
            #points_np = farthest_point_sample(points_np, NPOINTS)
            # random sample
            idx = np.random.choice(points_np.shape[0], NPOINTS, replace=False)
            points_np = points_np[idx, :]
            points_np = points_np[:NPOINTS, :]
        current_points = torch.from_numpy(points_np).to(torch.float32)
        """
        
        # for testing
        label_name = self.shape_names[label_idx] if not self.force_test_return else img_path # hack for retrieval experiment
        return current_points, label_idx, label_name
        
    def __len__(self):
        return len(self.datapath)

@DATASETS.register_module()
class ScanObjectNN(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.use_normals = getattr(config, 'USE_NORMALS', False)
        self.use_height = getattr(config, 'use_height', False)
        self.use_colored_pc = getattr(config, 'use_colored_pc', False)
        self.dataset_name = getattr(config, 'dataset_name', 'ScanObjectNN')

        h5_name = os.path.join(self.root, f'{self.subset}.h5')
        if not os.path.exists(h5_name):
             # The official dataset has different naming conventions
             if self.subset == 'train':
                 h5_name = os.path.join(self.root, 'h5_files', 'main_split_nobg', 'training_objectdataset.h5')
             elif self.subset == 'test':
                 h5_name = os.path.join(self.root, 'h5_files', 'main_split_nobg', 'test_objectdataset.h5')
             else:
                 raise FileNotFoundError(f"Cannot find ScanObjectNN h5 file for subset {self.subset}")

        print(f'Loading data from {h5_name}')
        try:
            with h5py.File(h5_name, 'r') as f:
                self.data = f['data'][:]
                self.label = f['label'][:]
        except FileNotFoundError:
            raise FileNotFoundError(f"ScanObjectNN data file not found at {h5_name}")

        print(f'The size of {self.subset} data is {len(self.data)}')

        # Load label names
        ordered_labels_path = os.path.join("./data", 'labels.json')
        try:
            with open(ordered_labels_path, 'r') as f:
                self.shape_names = json.load(f)[self.dataset_name]  # Use official NuScenes-Objects labels
        except Exception as e:
            raise ValueError(f"Error loading ordered label list from {ordered_labels_path}: {e}")

    def __len__(self):
        return self.data.shape[0]

    def _get_item(self, index):
        point_set = self.data[index]
        label = self.label[index]

        # Subsample if necessary, reproducibly
        if point_set.shape[0] > self.npoints:
            choice_seed = abs(hash(f"scanobjectnn_{self.subset}_{index}")) % (2**32)
            rng = np.random.default_rng(choice_seed)
            choice = rng.choice(point_set.shape[0], self.npoints, replace=False)
            point_set = point_set[choice, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            gravity_dim = 1
            height_array = point_set[:, gravity_dim:gravity_dim + 1] - point_set[:, gravity_dim:gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)

        return point_set, label

    def __getitem__(self, index):
        points_np, label = self._get_item(index)
        
        # Shuffle points during training, keep order for testing
        if self.subset == 'train':
            np.random.shuffle(points_np)

        label_name = self.shape_names[int(label)]
        points = torch.from_numpy(points_np).float()

        return points, label, label_name


@DATASETS.register_module()
class ModelNet(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset
        self.use_10k_pc = config.use_10k_pc
        self.use_colored_pc = config.use_colored_pc

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='ModelNet')

        if self.uniform:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                # make sure you have raw data in the path before you enable generate_from_raw_data=True.
                if self.generate_from_raw_data:
                    print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
                    self.list_of_points = [None] * len(self.datapath)
                    self.list_of_labels = [None] * len(self.datapath)

                    for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                        fn = self.datapath[index]
                        cls = self.classes[self.datapath[index][0]]
                        cls = np.array([cls]).astype(np.int32)
                        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                            print_log("uniformly sampled out {} points".format(self.npoints))
                        else:
                            point_set = point_set[0:self.npoints, :]

                        self.list_of_points[index] = point_set
                        self.list_of_labels[index] = cls

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.list_of_points, self.list_of_labels], f)
                else:
                    # no pre-processed dataset found and no raw data found, then load 8192 points dataset then do fps after.
                    self.save_path = os.path.join(self.root,
                                                  'modelnet%d_%s_%dpts_fps.dat' % (
                                                  self.num_category, split, 8192))
                    print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                    if not self.use_10k_pc:
                        print_log('since no exact points pre-processed dataset found and no raw data found, load 8192 pointd dataset first, if downsampling with fps to {} happens later, the speed is excepted to be slower due to fps...'.format(self.npoints), logger='ModelNet')
                    with open(self.save_path, 'rb') as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)

            else:
                print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        self.shape_names_addr = os.path.join(self.root, 'modelnet40_shape_names.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.shape_names = lines

        self.use_height = config.use_height
        
        if self.use_10k_pc and self.use_colored_pc:
            self.modelnet_10k_colored_pc_file = 'data/modelnet40_normal_resampled/modelnet40_colored_10k_pc.npy'
            self.modelnet_10k_rgb_data = np.load(self.modelnet_10k_colored_pc_file, allow_pickle=True)
            with open('data/modelnet40_normal_resampled/modelnet40_test_split_10k_colored.json', 'r') as f:
                self.cat_name = json.load(f)

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        if  self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)

        if self.use_10k_pc and self.use_colored_pc:
            point_set = self.modelnet_10k_rgb_data[index]['xyz']
            rgb_data = np.ones_like(point_set) * 0.4
            point_set = np.concatenate([point_set, rgb_data], axis=1)
            cat_name = self.cat_name[index]['category']
            label = [self.shape_names.index(cat_name)]
        elif self.use_colored_pc:
            rgb_data = np.ones_like(point_set) * 0.4
            point_set = np.concatenate([point_set, rgb_data], axis=1)

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label_name = self.shape_names[int(label)]

        return current_points, label, label_name

@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):

        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.id_map_addr = os.path.join(config.DATA_PATH, 'taxonomy.json')
        self.rendered_image_addr = config.IMAGE_PATH
        self.picked_image_type = ['', '_depth0001']
        self.picked_rotation_degrees = list(range(0, 360, 12))
        self.picked_rotation_degrees = [(3 - len(str(degree))) * '0' + str(degree) if len(str(degree)) < 3 else str(degree) for degree in self.picked_rotation_degrees]

        with open(self.id_map_addr, 'r') as f:
            self.id_map = json.load(f)

        self.prompt_template_addr = os.path.join('./data/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.synset_id_map = {}
        for id_dict in self.id_map:
            synset_id = id_dict["synsetId"]
            self.synset_id_map[synset_id] = id_dict

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line[len(taxonomy_id) + 1:].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)

        self.uniform = True
        self.augment = True
        self.use_caption_templates = False
        self.use_height = config.use_height

        if self.augment:
            print("using augmented point clouds.")

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        if self.uniform and self.sample_points_num < data.shape[0]:
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)

        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        captions = self.synset_id_map[sample['taxonomy_id']]['name']
        captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
        caption = random.choice(captions)
        captions = []
        tokenized_captions = []
        if self.use_caption_templates:
            for template in self.templates:
                caption = template.format(caption)
                captions.append(caption)
                tokenized_captions.append(self.tokenizer(caption))
        else:
            tokenized_captions.append(self.tokenizer(caption))

        tokenized_captions = torch.stack(tokenized_captions)

        picked_model_rendered_image_addr = self.rendered_image_addr + '/' +\
                                           sample['taxonomy_id'] + '-' + sample['model_id'] + '/'
        picked_image_name = sample['taxonomy_id'] + '-' + sample['model_id'] + '_r_' +\
                            str(random.choice(self.picked_rotation_degrees)) +\
                            random.choice(self.picked_image_type) + '.png'
        #picked_image_addr = picked_model_rendered_image_addr + picked_image_name
        picked_image_addr = self.rendered_image_addr + '/' + picked_image_name # workaround to work with the smaller ShapeNet55 triplets (depth images)
       
        try:
            image = pil_loader(picked_image_addr)
            image = self.train_transform(image)
        except:
            raise ValueError("image is corrupted: {}".format(picked_image_addr))

        #return sample['taxonomy_id'], sample['model_id'], tokenized_captions, data, image
        return sample['model_id'], tokenized_captions, data, image

    def __len__(self):
        return len(self.file_list)
    
@DATASETS.register_module()
class Objaverse_Lvis_Colored(data.Dataset):
    def __init__(self, config):
            
        self.nuscenes_median = 574

        self.npoints = config.npoints
        #self.npoints = 10000
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform

        self.lvis_list_addr = os.path.join(config.DATA_PATH, 'lvis.json')
        self.lvis_metadata_addr = os.path.join(config.DATA_PATH, 'objaverse_lvis_metadata.json')
        #self.lvis_list_addr = 'data/objaverse-lvis/lvis.json'
        #self.lvis_metadata_addr = 'data/objaverse-lvis/objaverse_lvis_metadata.json'

        with open(self.lvis_list_addr, 'r') as f:
            self.npy_file_map = json.load(f)

        self.file_list = list(self.npy_file_map.keys())

        with open(self.lvis_metadata_addr, 'r') as f:
            self.lvis_metadata = json.load(f)

        self.prompt_template_addr = 'data/templates.json'
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.sample_points_num = self.npoints
        self.cap_to_npoints = config.cap_to_npoints

        print_log(f'Objaverse lvis {len(self.file_list)} instances were loaded', logger='objaverse_lvis')

        self.permutation = np.arange(self.npoints)

        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = False
        self.use_color = config.use_colored_pc
        
        self.objaverse_lvis_path = 'data/objaverse-lvis'
        
        if self.use_color:
            print("Using colored point clouds from Objaverse-LVIS.")
        else:
            print("Using XYZ point clouds from Objaverse-LVIS.")

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):

        sample = self.file_list[idx]
        pc_addr = self.npy_file_map[sample]
        pc_addr = os.path.join(self.objaverse_lvis_path,self.npy_file_map[sample])
        data = np.load(pc_addr, allow_pickle=True)
        dict_data = data.item()
        xyz_data = dict_data['xyz']
        rgb_data = dict_data['rgb']

        data = self.pc_norm(xyz_data)
        if self.use_color:
            data = np.concatenate([data, rgb_data], axis=1)
            
        # resample to get nuscenes median many points
        #if data.shape[0] > self.nuscenes_median:
        #    indices = self.generator.choice(data.shape[0], self.nuscenes_median, replace=False)
        #    data = data[indices, :]
        
        if self.cap_to_npoints and data.shape[0] > self.npoints:
            object_seed = abs(hash(sample)) % (2**32)
            local_rng = np.random.Generator(np.random.PCG64(object_seed))
            
            indices = local_rng.choice(data.shape[0], self.npoints, replace=False)
            data = data[indices, :]
            
        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            
        data = torch.from_numpy(data).float()
        data = data.contiguous()

        name = self.lvis_metadata["value_to_key_mapping"][sample]
        label = self.lvis_metadata["key_to_id"][name]

        return data, label, name

    def __len__(self):
        return len(self.file_list)

import collections.abc as container_abcs
int_classes = int
#from torch._six import string_classes
from six import string_types as string_classes

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')

from collections.abc import Mapping
from torch.utils.data import default_collate

def customized_collate_fn(batch):
    is_validation = isinstance(batch, list) and len(batch[0]) == 3
    
    points_idx = 0 if is_validation else 2
    
    # discard batches with None images
    if isinstance(batch, list) and not is_validation:
        batch = [example for example in batch if example[3] is not None]
    
    elem = batch[0]
    elem_type = type(elem)
    
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        
        # PADDING LOGIC (if applicable)        
        batch_as_lists = [list(item) for item in batch]
        #max_pc_size = max([len(b[points_idx]) for b in batch_as_lists])
        max_pc_size = 8192 # BURNED IN NOW FOR THESIS
        
        for t in batch_as_lists:
            pc = t[points_idx]
            if len(pc) < max_pc_size:
                padding = torch.full(
                    (max_pc_size - len(pc), pc.size(1)), 
                    float('inf'),
                    dtype=pc.dtype, 
                    device=pc.device
                )
                t[points_idx] = torch.cat([pc, padding], dim=0)
                
        # Convert back to tuples and continue
        batch = [tuple(item) for item in batch_as_lists]
        
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]
        
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class Dataset_3D():
    def __init__(self, args, tokenizer, dataset_type, train_transform=None):
        if dataset_type == 'train':
            self.dataset_name = args.pretrain_dataset_name
        elif dataset_type == 'val':
            self.dataset_name = args.validate_dataset_name
        elif dataset_type == 'val_2' and args.validate_dataset_name_2 and args.validate_dataset_prompt_2:
            self.dataset_name = args.validate_dataset_name_2
        else:
            raise ValueError("not supported dataset type.")
        with open('./data/dataset_catalog.json', 'r') as f:
            self.dataset_catalog = json.load(f)
            self.dataset_usage = self.dataset_catalog[self.dataset_name]['usage']
            self.dataset_split = self.dataset_catalog[self.dataset_name][self.dataset_usage]
            self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        self.pretrain_dataset_prompt = args.pretrain_dataset_prompt
        self.validate_dataset_prompt = args.validate_dataset_prompt
        if 'colored' in args.model.lower():
            self.use_colored_pc = True
        else:
            self.use_colored_pc = False
        if args.npoints == 10000:
            self.use_10k_pc = True
        else:
            self.use_10k_pc = False
        self.build_3d_dataset(args, self.dataset_config_dir)

    def build_3d_dataset(self, args, config):
        config = cfg_from_yaml_file(config)
        config.tokenizer = self.tokenizer
        config.train_transform = self.train_transform
        config.pretrain_dataset_prompt = self.pretrain_dataset_prompt
        config.validate_dataset_prompt = self.validate_dataset_prompt
        config.args = args
        config.use_height = args.use_height
        config.npoints = args.npoints
        config.cap_to_npoints = "pointbert" in args.model.lower()
        config.use_colored_pc = self.use_colored_pc
        config.use_10k_pc = self.use_10k_pc
        config.sim_occlusion = args.sim_occlusion
        config.dataset_name = self.dataset_name
        config.excluded_classes = args.excluded_classes
        #config.ignore_normals = args.ignore_normals
        config_others = EasyDict({'subset': self.dataset_split, 'whole': True})
        self.dataset = build_dataset_from_cfg(cfg=config, default_args=config_others)
