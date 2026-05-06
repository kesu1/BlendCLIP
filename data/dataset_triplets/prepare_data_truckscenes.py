from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud, Box
from truckscenes.utils.geometry_utils import points_in_box, transform_matrix
from truckscenes.utils.splits import create_splits_scenes
from truckscenes.utils.geometry_utils import view_points, BoxVisibility
from concurrent.futures import ThreadPoolExecutor
import threading
from pyquaternion import Quaternion
import numpy as np
import os
import h5py
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import psutil
from tqdm import tqdm
import json

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def __len__(self):
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()

# All available LiDAR sensor names in TruckScenes
LIDAR_NAMES_TRUCKSCENES = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 
                           'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR']

GENERAL_TO_DETECTION = {
    # valid evaluation classes
    "animal": "animal",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic cone",
    "static_object.traffic_sign": "traffic sign",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "other vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
    "vehicle.other": "other vehicle",
    "vehicle.ego_trailer": "trailer",

    # void classes
    "human.pedestrian.personal_mobility": "void",
    "human.pedestrian.stroller": "void",
    "human.pedestrian.wheelchair": "void",
    "movable_object.debris": "void",
    "movable_object.pushable_pullable": "void",
    "static_object.bicycle_rack": "void",
    "vehicle.emergency.ambulance": "void",
    "vehicle.emergency.police": "void",
    "vehicle.train": "void",
}

def visualize_point_cloud(points):
    """
    Visualize a point cloud using matplotlib with color and fixed axis lengths.
    
    Parameters:
        points (np.ndarray): A (N, 6) numpy array representing the point cloud (x, y, z, r, g, b).
    """
    # Create a new figure and a 3D Axes.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract X, Y, Z coordinates and colors from the point cloud.
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    colors = points[:, 3:6]  # Use saved color (r, g, b)
    
    # Create a scatter plot. Adjust s (size) as needed.
    scatter = ax.scatter(x, y, z, s=1, c=colors)
    
    # Set axis labels.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Fix axis lengths to avoid distortion.
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Display the plot.
    plt.show()
    
def delta_pos_rot(ann_from, ann_to):
        # Get position difference (in m)
        pos_from = np.array(ann_from['translation'])
        pos_to = np.array(ann_to['translation'])
        pos_diff = pos_to - pos_from
        
        # Get orientation difference (in quat)
        rot_from = Quaternion(ann_from['rotation'])
        rot_to = Quaternion(ann_to['rotation'])
        rot_diff = rot_to * rot_from.inverse
        
        return pos_diff, rot_diff    
    
def collect_camera_data(dataset, sample_token, ann_rec, image_cache=None):
    """
    Collect camera data needed for point cloud colorizing.
    """
     # Check visibility level first
    visibility_token = ann_rec['visibility_token']
    visibility_level = dataset.get('visibility', visibility_token)['level']
    if visibility_level < 4:  # Only process instances with highest visibility
        return []
    
    sample_rec = dataset.get('sample', sample_token)
    all_imgs_data = []
    
    # Get all camera names
    cams = [k for k in sample_rec['data'].keys() if 'CAMERA' in k]
    
    for cam in cams:
        try:
            sample_data_token = sample_rec['data'][cam]
            
            # Get data path and camera intrinsic
            data_path, _, cam_intrinsic = dataset.get_sample_data(
                sample_data_token,
                box_vis_level=BoxVisibility.ALL,
                selected_anntokens=[ann_rec['token']]
            )
            
            # Skip if no valid path
            if not data_path:
                continue
                
            # Get camera record
            cam_sd_rec = dataset.get('sample_data', sample_data_token)
            
            try:
                if image_cache:
                    # Load image with caching
                    im = image_cache.get(data_path) if image_cache else None
                    if im is None:
                        im = Image.open(data_path)
                        if image_cache:
                            image_cache.put(data_path, im)
                else:
                    # Load image without caching
                    im = Image.open(data_path)
            
                im_np = np.array(im)
                
                # Get camera pose
                cam_calib = dataset.get('calibrated_sensor', cam_sd_rec['calibrated_sensor_token'])
                cam_pose = dataset.get('ego_pose', cam_sd_rec['ego_pose_token'])
                T_cam = transform_matrix(cam_calib['translation'], 
                                        Quaternion(cam_calib['rotation']), 
                                        inverse=False)
                T_ego = transform_matrix(cam_pose['translation'], 
                                        Quaternion(cam_pose['rotation']), 
                                        inverse=False)
                T_camera = T_ego @ T_cam  # sensor -> global
                T_global_to_cam = np.linalg.inv(T_camera)
                
                # Store data
                cam_data = {
                    'image_np': im_np,
                    'T_global_to_cam': T_global_to_cam,
                    'cam_intrinsic': cam_intrinsic,
                    'ann_rec': ann_rec
                }
                
                all_imgs_data.append(cam_data)
                
            except (PIL.UnidentifiedImageError, OSError) as img_err:
                print(f"Warning: Could not load image {data_path}: {str(img_err)}")
                continue
                
        except Exception as e:
            print(f"Error processing camera {cam} for sample {sample_token}: {str(e)}")
            continue
    
    return all_imgs_data

def colorize_point_cloud(points, all_imgs_data, pc_ann_rec):
    """
    Colorize point cloud with temporal motion compensation.
    """ 
    N = points.shape[0]
    color_sum = np.zeros((N, 3), dtype=np.float32)
    color_count = np.zeros(N, dtype=np.int32)

    for cam_data in all_imgs_data:
        T_global_to_cam = cam_data['T_global_to_cam']
        cam_intrinsic = cam_data['cam_intrinsic']
        img_ann_rec = cam_data['ann_rec']
        im_np = cam_data['image_np']
        height, width, _ = im_np.shape
        
        pos_diff, rot_diff = delta_pos_rot(ann_from=pc_ann_rec, ann_to=img_ann_rec)
        pc_box_center = pc_ann_rec['translation']

        adjusted_points = points.copy()
        # Apply rotation
        if rot_diff.angle > 1e-6:
            adjusted_points[:, :3] -= pc_box_center
            rotation_matrix = rot_diff.rotation_matrix
            adjusted_points[:, :3] = adjusted_points[:, :3] @ rotation_matrix.T
            adjusted_points += pc_box_center
        # Apply translation
        if np.linalg.norm(pos_diff) > 1e-6:
            adjusted_points += pos_diff
    
        # Prepare homogeneous coordinates for cropped points (shape: (4, N))
        pts_hom = np.hstack((adjusted_points, np.ones((N, 1)))).T
        pts_cam = T_global_to_cam @ pts_hom
        pts_cam_xyz = pts_cam[:3, :]
        
        proj = view_points(pts_cam_xyz, cam_intrinsic, normalize=True)
                    
        # Determine valid points: in front of camera and within image bounds.
        valid = (pts_cam[2, :] > 0) & (proj[0, :] >= 0) & (proj[0, :] < width) & (proj[1, :] >= 0) & (proj[1, :] < height)
        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            continue
        
        u = np.round(proj[0, valid_idx]).astype(np.int32)
        v = np.round(proj[1, valid_idx]).astype(np.int32)
        
        # Clip values to ensure they stay within image boundaries
        u = np.clip(u, 0, width - 1)
        v = np.clip(v, 0, height - 1)
        
        colors = im_np[v, u, :]  # shape (num_valid, 3)
        color_sum[valid_idx] += colors.astype(np.float32)
        color_count[valid_idx] += 1
                
    # Finalize colors: average if visible, otherwise set to white.
    final_colors = np.where(color_count[:, None] > 0,
                            color_sum / color_count[:, None],
                            np.array([255, 255, 255], dtype=np.float32))
    final_colors = final_colors / 255.0  # normalize to [0,1]
    # Append color to each point so that dimension becomes (x,y,z,r,g,b)
    colored_points = np.hstack((points, final_colors))
    return colored_points


def load_and_fuse_lidar(dataset, sample_token):
    """
    Load and fuse LiDAR data from all sensors for a given sample.
    
    Args:
        dataset: TruckScenes dataset object
        sample_token: Token for the sample
        
    Returns:
        fused_points: (3, N) array of fused point cloud coordinates
    """
    sample_record = dataset.get('sample', sample_token)
    fused_points_list = []
    
    for lidar_name in LIDAR_NAMES_TRUCKSCENES:
        try:
            lidar_token = sample_record['data'][lidar_name]
            lidar_record = dataset.get('sample_data', lidar_token)
            
            # Load the point cloud from file
            #pcl_path = os.path.join(dataset.dataroot, lidar_record['filename'])
            
            original_filename = lidar_record['filename']
            path_parts = original_filename.split('/')
            path_parts[-1] = "trainval_" + path_parts[-1]  # Add prefix to last part (filename)
            filename_with_prefix = '/'.join(path_parts)
            pcl_path = os.path.join(dataset.dataroot, filename_with_prefix)
            
            pc = LidarPointCloud.from_file(pcl_path)
            
            # Get transforms (sensor → ego → global)
            calib_rec = dataset.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
            ego_pose_rec = dataset.get('ego_pose', lidar_record['ego_pose_token'])
            sensor2ego = transform_matrix(calib_rec['translation'],
                                        Quaternion(calib_rec['rotation']),
                                        inverse=False)
            ego2global = transform_matrix(ego_pose_rec['translation'],
                                        Quaternion(ego_pose_rec['rotation']),
                                        inverse=False)
            T = ego2global @ sensor2ego  # sensor -> global
            
            # Transform the points (using homogeneous coordinates)
            points = pc.points[:3, :]  # (3, N)
            ones = np.ones((1, points.shape[1]), dtype=points.dtype)
            points_hom = np.vstack((points, ones))
            points_global = (T @ points_hom)[:3, :]
            fused_points_list.append(points_global)
        except Exception as e:             # Continue to the next LiDAR sensor
            print(f"Error processing PCD file {pcl_path}: {str(e)}")
            continue
    
    if fused_points_list:
        fused_points = np.concatenate(fused_points_list, axis=1)
    else:
        fused_points = np.empty((3, 0))
    
    return fused_points

def process_instance(dataset, instance, results_dir, pts_tresh, scene_names=None,
                     lidar_cache=None, sample_cache=None, image_cache=None):
    """
    Process a single instance to extract cropped point clouds.

    Args:
        dataset: TruckScenes dataset object
        instance: Instance dictionary
        results_dir: Directory to save results
        pts_tresh: Minimum number of points required
        scene_names: List of scene names to include (for split filtering)
        lidar_cache: Cache for LiDAR point clouds
        sample_cache: Cache for sample records
    """
    instance_token = instance['token']
    
    # Skip "vehicle.other" category
    if dataset.get('category', instance['category_token'])["name"] == "vehicle.other":
        with open(os.path.join(results_dir, "skipped.txt"), "a") as f:
            f.write(f"{instance_token}\n")
        #print(f"Skipped instance {instance_token}: 'vehicle.other' category.")
        return False
    
    ann_tokens = dataset.field2token('sample_annotation', 'instance_token', instance_token)
    
    all_cropped_pcs = []        # list of (pc_np, pc_id)
    all_pcs_ann_rec = []        # list of all point clouds' annotation records
    #all_imgs_data = []          # List of all images data for colorizing
    
    pc_id_counter = 0

    for ann_token in ann_tokens:
        ann_rec = dataset.get('sample_annotation', ann_token)
        sample_token = ann_rec['sample_token']
        
        # Cache sample records to avoid duplicate I/O calls
        if sample_cache:
            sample_rec = sample_cache.get(sample_token)
            if sample_rec is None:
                sample_rec = dataset.get('sample', sample_token)
                sample_cache.put(sample_token, sample_rec)
        else:
            sample_rec = dataset.get('sample', sample_token)
            
        # Check if scene belongs to the specified split, skip if not
        scene_token = sample_rec.get('scene_token')
        scene_rec = dataset.get('scene', scene_token)
        if scene_rec['name'] not in scene_names:
            continue
        
        # Skip if annotation doesn't have enough LiDAR points
        #if ann_rec['num_lidar_pts'] < pts_tresh:
        #    continue

        """
         # Collect camera images data for this annotation
        imgs_data = collect_camera_data(dataset, sample_token, ann_rec, image_cache)
        if imgs_data:
            all_imgs_data.extend(imgs_data)
        #print(f"Memory usage after cam collection: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        """
            
        if lidar_cache:
            # Get fused LiDAR data with caching
            lidar_pts = lidar_cache.get(sample_token)
            if lidar_pts is None:
                lidar_pts = load_and_fuse_lidar(dataset, sample_token)
                lidar_cache.put(sample_token, lidar_pts)
        else:
            lidar_pts = load_and_fuse_lidar(dataset, sample_token)
        
        # Crop points inside the bounding box
        bbox = Box(center=ann_rec['translation'],
                   size=ann_rec['size'],
                   orientation=Quaternion(ann_rec['rotation']))
        in_mask = points_in_box(box=bbox, points=lidar_pts)
        
        # Only keep if it meets the threshold
        if np.sum(in_mask) >= pts_tresh:
            cropped_lidar_pts = lidar_pts[:, in_mask].T  # (N, 3)
            all_cropped_pcs.append((cropped_lidar_pts, pc_id_counter))
            all_pcs_ann_rec.append(ann_rec)  # Store annotation record
            pc_id_counter += 1

    # Save data if we have point clouds
    if len(all_cropped_pcs) > 0:
        # Create the directory structure
        pc_dir = os.path.join(results_dir, 'pc')
        os.makedirs(pc_dir, exist_ok=True)
        
        # Save point clouds to HDF5
        hdf5_path = os.path.join(pc_dir, f"{instance_token}.hdf5")
        with h5py.File(hdf5_path, "w") as f:
            #for pc_array, pc_id in all_cropped_pcs:
            #    dataset_name = f"pc_{pc_id}"
            #    f.create_dataset(dataset_name, data=pc_array, compression="gzip")
            for i, (pc_array, pc_id) in enumerate(all_cropped_pcs):
                """
                # Color the point cloud
                colored_pc = colorize_point_cloud(
                    pc_array, 
                    all_imgs_data,
                    pc_ann_rec=all_pcs_ann_rec[i]
                )
                #print(f"Memory usage after coloring: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
                """

                
                # Center the points around their CoM (like in NuScenes)
                center = np.mean(pc_array[:, :3], axis=0)
                pc_array[:, :3] -= center
                
                #visualize_point_cloud(colored_pc)  # Visualize the point cloud
                
                dataset_name = f"pc_{pc_id}"
                f.create_dataset(dataset_name, data=pc_array, compression="gzip")
        
        # Save class labels
        category_name = dataset.get('category', instance['category_token'])["name"]
        label = GENERAL_TO_DETECTION.get(category_name, category_name)
        with open(os.path.join(results_dir, "labels.txt"), "a") as f:
            f.write(f"{instance_token}:{label}\n")
        #print(f"Saved {len(all_cropped_pcs)} point clouds for instance {instance_token}")
        return True
    else:
        # No usable data for this instance
        with open(os.path.join(results_dir, "skipped.txt"), "a") as f:
            f.write(f"{instance_token}\n")
        #print(f"Skipped instance {instance_token} due to insufficient data.")
        return False

def get_gt_eval_boxes(dataset):
    from truckscenes.eval.common.loaders import (
        add_center_dist,
        filter_eval_boxes,
        load_gt)
    
    #from nuscenes.eval.common.data_classes import EvalBox
    from truckscenes.eval.detection.data_classes import DetectionBox
    from truckscenes.eval.common.config import DetectionConfig
    
    gt_boxes = load_gt(dataset, "val", DetectionBox, verbose=True)
    gt_boxes = add_center_dist(dataset, gt_boxes)
    
    with open("detection_cvpr_2024_truckscenes.json", 'r') as f:
        cfg_data = json.load(f)
    cfg = DetectionConfig.deserialize(cfg_data)

    gt_boxes = filter_eval_boxes(dataset, gt_boxes, cfg.class_range, verbose=True)
    
    return gt_boxes

def process_eval_boxes(dataset, eval_boxes, results_dir, pts_tresh, point_cloud_cache=None):
    
    instance_boxes = {}
        
    for sample_token in eval_boxes.sample_tokens:    
        for box in eval_boxes[sample_token]:
            instance_token = dataset.get('sample_annotation', box.ann_token)['instance_token']
            if instance_token not in instance_boxes:
                instance_boxes[instance_token] = []
            instance_boxes[instance_token].append(box)
            
    print(f"Found {len(instance_boxes)} instances with evaluation boxes.")
    
    for instance_token, boxes in tqdm(instance_boxes.items(), desc="Processing instances"):
        all_cropped_pcs = []        # list of (pc_np, pc_id)
        pc_id_counter = 0
        
        for box in boxes:
            sample_token = box.sample_token
            
            # Use motion-compensated point clouds
            try:
                if point_cloud_cache:
                    # Get fused LiDAR data with caching
                    lidar_pts = point_cloud_cache.get(sample_token)
                    if lidar_pts is None:
                        lidar_pts = load_and_fuse_lidar(dataset, sample_token)
                        point_cloud_cache.put(sample_token, lidar_pts)
                else:
                    lidar_pts = load_and_fuse_lidar(dataset, sample_token)
                                
                bbox = Box(
                    center=box.translation,
                    size=box.size,
                    orientation=Quaternion(box.rotation))
                
                bbox_mask = points_in_box(box=bbox, points=lidar_pts)
                if np.sum(bbox_mask) >= pts_tresh:
                    lidar_pts = lidar_pts[:, bbox_mask]

                    cropped_lidar_pts = lidar_pts.T  # (N, 3)
                    all_cropped_pcs.append((cropped_lidar_pts, pc_id_counter))      
                    pc_id_counter += 1
                    
            except Exception as e:
                print(f"Error processing box {box.ann_token} in instance {instance_token}: {e}")
                continue

        os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        if len(all_cropped_pcs) > 0:
            # Save data after processing all boxes for this instance
                
            # Save the point clouds
            pc_dir = os.path.join(results_dir, 'pc')
            os.makedirs(pc_dir, exist_ok=True)

            hdf5_path = os.path.join(pc_dir, f"{instance_token}.hdf5")
            with h5py.File(hdf5_path, "w") as f:
                for pc_array, pc_id in all_cropped_pcs:
                    dset_name = f"pc_{pc_id}"
                        
                    # Center the points around their CoM
                    center = np.mean(pc_array[:, :3], axis=0)
                    pc_array[:, :3] -= center
                        
                    f.create_dataset(dset_name, data=pc_array, compression="gzip")
                        
            # Save label (textual description)
            instance = dataset.get('instance', instance_token)
            instance_name = dataset.get('category', instance['category_token'])["name"]
            label = GENERAL_TO_DETECTION.get(instance_name, instance_name)
                
            labels_path = os.path.join(results_dir, "labels.txt")
            with open(labels_path, "a", encoding="utf-8") as f:
                f.write(f"{instance_token}:{label}\n")        
                    
        #print(f"Saved {len(all_cropped_pcs)} point clouds for instance {instance_token} at {hdf5_path}")

def main(args):
    # Initialize the dataset
    trucksc = TruckScenes(version=args.version,
                          dataroot=args.dataroot,
                          verbose=True)
    
        # if the dataset loaded is v1.0-trainval
    if args.version == 'v1.0-trainval':
        scene_splits = create_splits_scenes(verbose=True)
        scene_names = scene_splits[args.split]
    
    results_dir = os.path.join(args.output, args.split)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize caches
    lidar_cache = LRUCache(capacity=2000)   # Cache for LiDAR point clouds
        
    boxes = get_gt_eval_boxes(trucksc)
    process_eval_boxes(trucksc, boxes, results_dir, args.pts_tresh, lidar_cache)
    
    print(f"\nFinished processing the specified split: {args.split}!\n")
                
    #print(f"\nFinished processing! Successfully processed {success_count} new instances.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TruckScenes dataset.")
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='TruckScenes dataset version')
    parser.add_argument('--split', type=str, default='val', help='Dataset split (train/val/test/all)')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to TruckScenes dataset root directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--pts_tresh', type=int, default=0, help='Minimum number of points in a bounding box to be considered valid')
    
    args = parser.parse_args()
    
    main(args)