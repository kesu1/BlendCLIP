import os
import json
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import points_in_box, transform_matrix, view_points, BoxVisibility
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
import open3d as o3d
import argparse
from collections import OrderedDict
from tqdm import tqdm

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

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        else:
            # Move to end = recently used
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            # Move to end = recently used
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove least recently used item
            self.cache.popitem(last=False)
    
    def __len__(self):
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()

LIDAR_NAME_NUSCENES = ['LIDAR_TOP']  # Only one LIDAR in nuScenes

# Official nuScenes object detection classes
GENERAL_TO_DETECTION = {
    # void / ignore classes
    "animal": "void",
    "human.pedestrian.personal_mobility": "void",
    "human.pedestrian.stroller": "void",
    "human.pedestrian.wheelchair": "void",
    "movable_object.debris": "void",
    "movable_object.pushable_pullable": "void",
    "static_object.bicycle_rack": "void",
    "vehicle.emergency.ambulance": "void",
    "vehicle.emergency.police": "void",

    # valid detection classes
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

def visualize_point_cloud(points):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    colors = points[:, 3:6]  # Use saved color (r, g, b)
    scatter = ax.scatter(x, y, z, s=1, c=colors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def visualize_point_cloud_o3d(points, show_axes=False, axis_size=1.0, grid_size=1.0, grid_density=5, point_size=25.0):
    """
    Visualize a colored point cloud using Open3D with coordinate axes for scale.
    
    Parameters:
        points (np.ndarray): A (N, 6) numpy array representing the point cloud.
                            Each row is (x, y, z, r, g, b) with colors normalized to [0,1].
        show_axes (bool): Whether to show coordinate axes
        axis_size (float): Size of the coordinate frame
        grid_size (float): Size of the grid (extends from -grid_size to +grid_size)
        grid_density (int): Number of grid lines in each direction
        point_size (float): Size of the points in the visualization
    """
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set points (xyz coordinates)
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Set colors (rgb values)
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    
    # Items to visualize
    visualization_items = [pcd]
    
    if show_axes:
        # Create grid lines for better scale reference
        grid_points = []
        grid_lines = []
        grid_colors = []
        line_idx = 0
        
        # Calculate step size
        step = (2 * grid_size) / grid_density
        
        # Create grid on XZ plane (ground plane)
        for i in range(grid_density + 1):
            x = -grid_size + i * step
            # X axis lines
            grid_points.extend([[x, 0, -grid_size], [x, 0, grid_size]])
            grid_lines.append([line_idx, line_idx + 1])
            grid_colors.append([0.5, 0.5, 0.5])  # Light gray
            line_idx += 2
            
            # Z axis lines
            z = -grid_size + i * step
            grid_points.extend([[-grid_size, 0, z], [grid_size, 0, z]])
            grid_lines.append([line_idx, line_idx + 1])
            grid_colors.append([0.5, 0.5, 0.5])  # Light gray
            line_idx += 2
        
        # Create a LineSet for the grid
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(grid_points),
            lines=o3d.utility.Vector2iVector(grid_lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(grid_colors)
        visualization_items.append(line_set)
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size, origin=[0, 0, 0]
        )
        visualization_items.append(coordinate_frame)
    
    # Create visualizer with custom settings
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    for item in visualization_items:
        vis.add_geometry(item)
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = point_size  # Set the point size
    render_option.background_color = np.array([1, 1, 1])  # White background for better visibility
    
    # Set camera position
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


def load_lidar(dataset, sample_token):
    sample_rec = dataset.get('sample', sample_token)
    fused_points_list = []

    lidar_name = LIDAR_NAME_NUSCENES[0] # TODO: accquire LIDAR names similar to the cameras' case
    lidar_token = sample_rec['data'][lidar_name]
    lidar_record = dataset.get('sample_data', lidar_token)

    pc_path = os.path.join(dataset.dataroot, lidar_record['filename'])
    pc = LidarPointCloud.from_file(pc_path)

    calib_rec = dataset.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
    ego_pose_rec = dataset.get('ego_pose', lidar_record['ego_pose_token'])
    sensor2ego = transform_matrix(calib_rec['translation'],
                                  Quaternion(calib_rec['rotation']),
                                  inverse=False)
    ego2global = transform_matrix(ego_pose_rec['translation'],
                                  Quaternion(ego_pose_rec['rotation']),
                                  inverse=False)
    T = ego2global @ sensor2ego  # sensor -> global

    points = pc.points[:3, :]
    ones = np.ones((1, points.shape[1]), dtype=points.dtype)
    points_hom = np.vstack((points, ones))
    points_global = (T @ points_hom)[:3, :]
    fused_points_list.append(points_global)

    if fused_points_list:
        fused_points = np.concatenate(fused_points_list, axis=1)
    else:
        fused_points = np.empty((3, 0))

    return fused_points

def load_and_fuse_lidar_motion_compensated(dataset, sample_token, ann_rec, max_sweeps=10,
                                           point_cloud_cache={}, sweep_cache={}):
    """
    Load and aggregate point clouds with motion compensation for individual points.
    Uses the object's velocity to adjust points from different time frames.
    Uses the previous sweeps to adjust the points to the reference time.
    """
    sample_rec = dataset.get('sample', sample_token)
    fused_points_list = []
    fused_times_list = []  # Track time of each point
     
    # Get velocity from dataset method (in m/s)
    velocity = dataset.box_velocity(ann_rec['token'])
    velocity = np.array(velocity)
    
    
    # Get previous annotation orientation if available
    prev_orientation = None
    prev_timestamp = None
    prev_ann_token = dataset.get('sample_annotation', ann_rec['token'])['prev']
    
    if prev_ann_token != "":
        prev_ann = dataset.get('sample_annotation', prev_ann_token)
        prev_orientation = Quaternion(prev_ann['rotation'])
        prev_sample = dataset.get('sample', prev_ann['sample_token'])
        prev_timestamp = prev_sample['timestamp']
        prev_velocity = dataset.box_velocity(prev_ann['token'])
        prev_velocity = np.array(prev_velocity)
    
    # Process reference sweep (current time)
    lidar_name = LIDAR_NAME_NUSCENES[0]
    ref_lidar_token = sample_rec['data'][lidar_name]
    ref_sd_rec = dataset.get('sample_data', ref_lidar_token)
    ref_timestamp = ref_sd_rec['timestamp']
    ref_orientation = Quaternion(ann_rec['rotation'])
    
    # Get reference point cloud    
    ref_calib = dataset.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_pose = dataset.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_pc_path = os.path.join(dataset.dataroot, ref_sd_rec['filename'])
    
    # Cache reference point cloud
    ref_pc = point_cloud_cache.get(ref_pc_path)
    if ref_pc is None:
        ref_pc = LidarPointCloud.from_file(ref_pc_path)
        point_cloud_cache.put(ref_pc_path, ref_pc)
    
    # Transform to global frame
    sensor2ego_ref = transform_matrix(ref_calib['translation'], Quaternion(ref_calib['rotation']))
    ego2global_ref = transform_matrix(ref_pose['translation'], Quaternion(ref_pose['rotation']))
    T_ref = ego2global_ref @ sensor2ego_ref
    
    points = ref_pc.points[:3, :]
    ones = np.ones((1, points.shape[1]), dtype=points.dtype)
    points_hom = np.vstack((points, ones))
    points_global = (T_ref @ points_hom)[:3, :]
    
    # Create reference box
    ref_box = Box(
        center=ann_rec['translation'],
        size=ann_rec['size'],
        orientation=ref_orientation
    )
    
    # Filter points from reference sweep
    in_mask = points_in_box(box=ref_box, points=points_global)
    if np.sum(in_mask) > 0:
        fused_points_list.append(points_global[:, in_mask])
        # Set time=0 for reference sweep points (current time)
        fused_times_list.append(np.zeros(np.sum(in_mask)))
    
    # Use current frame only if no reliable velocity is available
    # Most likely the first annotation of the instance
    if np.any(np.isnan(velocity)):
        return points_global, fused_times_list
    
    # Process previous sweeps
    curr_sd_rec = ref_sd_rec
    for _ in range(max_sweeps-1):
        if curr_sd_rec['prev'] == "":
            break
            
        # Get previous sweep data
        curr_sd_rec = dataset.get('sample_data', curr_sd_rec['prev'])
        curr_timestamp = curr_sd_rec['timestamp']
        time_diff_sec = (ref_timestamp - curr_timestamp) / 1e6  # Convert to seconds
        
        # Load point cloud
        curr_calib = dataset.get('calibrated_sensor', curr_sd_rec['calibrated_sensor_token'])
        curr_pose = dataset.get('ego_pose', curr_sd_rec['ego_pose_token'])
        curr_pc_path = os.path.join(dataset.dataroot, curr_sd_rec['filename'])
        
        # Cache loaded sweeps
        curr_pc = sweep_cache.get(curr_pc_path)
        if curr_pc is None:
            curr_pc = LidarPointCloud.from_file(curr_pc_path)
            sweep_cache.put(curr_pc_path, curr_pc)
        
        # Transform to global
        sensor2ego_curr = transform_matrix(curr_calib['translation'], Quaternion(curr_calib['rotation']))
        ego2global_curr = transform_matrix(curr_pose['translation'], Quaternion(curr_pose['rotation']))
        T_curr = ego2global_curr @ sensor2ego_curr
        
        points = curr_pc.points[:3, :]
        ones = np.ones((1, points.shape[1]), dtype=points.dtype)
        points_hom = np.vstack((points, ones))
        points_global_curr = (T_curr @ points_hom)[:3, :]
        
        # Calculate adjusted box position for previous sweep
        ## Interpolate velocity based on temporal position between annotations
        interp_velocity = velocity  # Default to current velocity

        if prev_ann_token != "" and not np.any(np.isnan(velocity)) and not np.any(np.isnan(prev_velocity)):
            # Calculate time differences
            ann_time_diff = (ref_timestamp - prev_timestamp) / 1e6  # Time between annotations
            sweep_time_diff = (ref_timestamp - curr_timestamp) / 1e6  # Time between reference and this sweep
            
            if abs(ann_time_diff) > 1e-6:
                # Calculate interpolation factor based on temporal position
                # 0 = at current annotation, 1 = at previous annotation
                factor = sweep_time_diff / ann_time_diff
                
                # Clamp factor to valid range [0, 1] for interpolation
                # (allows extrapolation for sweeps outside the annotation range)
                # factor = np.clip(factor, 0, 1)  # Uncomment to prevent extrapolation
                
                # Linear interpolation between velocities
                interp_velocity = velocity * (1 - factor) + prev_velocity * factor

        position_offset = -interp_velocity * time_diff_sec
        adjusted_center = ref_box.center + position_offset
        
        # Calculate adjusted orientation if we have previous annotation data
        adjusted_orientation = ref_orientation
        if prev_orientation is not None and prev_timestamp is not None:
            # Time difference between annotations
            ann_time_diff = (ref_timestamp - prev_timestamp) / 1e6
            
            if abs(ann_time_diff) > 1e-6:
                # Interpolation factor based on sweep time relative to annotation time
                if curr_timestamp <= prev_timestamp:
                    # If sweep is earlier than prev_annotation, extrapolate further back
                    factor = 1.0 + (prev_timestamp - curr_timestamp) / (ref_timestamp - prev_timestamp)
                else:
                    # If sweep is between annotations, interpolate
                    factor = (ref_timestamp - curr_timestamp) / (ref_timestamp - prev_timestamp)
                
                # Spherical linear interpolation between the two orientations
                adjusted_orientation = Quaternion.slerp(ref_orientation, prev_orientation, factor)
        
        # Create motion-adjusted box for this sweep
        adjusted_box = Box(
            center=adjusted_center, 
            size=ref_box.wlh,
            orientation=adjusted_orientation,
            #orientation=ref_box.orientation,
        )
        
        # Filter points using adjusted box
        in_mask = points_in_box(box=adjusted_box, points=points_global_curr)
        if np.sum(in_mask) > 0:
            selected_points = points_global_curr[:, in_mask]

            # Rotate points to align with adjusted box orientation
            delta_rotation = ref_orientation * adjusted_orientation.inverse
            
            if delta_rotation.angle > 1e-6:
                selected_points -= adjusted_center.reshape(3, 1)
                
                R_delta = delta_rotation.rotation_matrix
                selected_points = R_delta @ selected_points

                selected_points += adjusted_center.reshape(3, 1)
            
            # Compensate for motion using interpolated velocity
            compensated_points = selected_points + (interp_velocity * time_diff_sec).reshape(3, 1)
            
            fused_points_list.append(compensated_points)
            fused_times_list.append(np.ones(np.sum(in_mask)) * time_diff_sec)
    
    # Combine all filtered points
    if fused_points_list:
        fused_points = np.concatenate(fused_points_list, axis=1)
        fused_times = np.concatenate(fused_times_list)
    else:
        fused_points = np.empty((3, 0))
        fused_times = np.empty(0)
    
    return fused_points, fused_times

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

def process_instance(dataset, instance, results_dir, pts_tresh, scene_names=None,
                     point_cloud_cache=None, sweep_cache=None, sample_cache=None, image_cache=None, scene_cache=None):
    """
    Process a single instance to extract cropped images and point clouds.
    We only save images and point clouds if there's at least one of each.

    The final file structure is:
        results_dir/images/instanceid.hdf5      # HDF5 with all images for that instance
        results_dir/pc/instanceid.hdf5          # HDF5 with all point clouds for that instance
    """

    instance_token = instance['token']
    ann_tokens = dataset.field2token('sample_annotation', 'instance_token', instance_token)
    
    #all_imgs_data = []          # List of all images data regarding the instance
    #all_cropped_images = []     # list of (img_np, image_id)
    
    all_cropped_pcs = []        # list of (pc_np, pc_id)
    all_pcs_ann_rec = []        # list of all point clouds' annotation records

    #image_id_counter = 0
    pc_id_counter = 0

    for ann_token in ann_tokens:
        ann_rec = dataset.get('sample_annotation', ann_token)
        sample_token = ann_rec['sample_token']
        
        # Cache sample records ans scene records
        sample_rec = sample_cache.get(sample_token)
        if sample_rec is None:
            sample_rec = dataset.get('sample', sample_token)
            sample_cache.put(sample_token, sample_rec)
            
        scene_token = sample_rec['scene_token']
        scene_rec = scene_cache.get(scene_token)
        if scene_rec is None:
            scene_rec = dataset.get('scene', scene_token)
            scene_cache.put(scene_token, scene_rec)
            
        # SKIP annotations from scenes not in our split    
        if scene_names is not None:
            scene_rec = dataset.get('scene', sample_rec['scene_token'])
            if scene_rec['name'] not in scene_names:
                continue
    
        """
        # IMAGE PROCESSING
        # Consider only the highest visibility (80-100%)
        if int(ann_rec['visibility_token']) == 4:
            cams = [k for k in sample_rec['data'].keys() if 'CAM' in k]
            for cam in cams:
                cam_token = sample_rec['data'][cam]
                cam_sd_rec = dataset.get('sample_data', cam_token)
                
                data_path, boxes, cam_intrinsic = dataset.get_sample_data(
                    cam_token,
                    box_vis_level=BoxVisibility.ALL,
                    selected_anntokens=[ann_token])

                if boxes:
                    im = image_cache.get(data_path)
                    if im is None:
                        im = Image.open(data_path)
                        image_cache.put(data_path, im)

                    width, height = im.size
                    
                    corners = np.stack([box.corners() for box in boxes], axis=0)
                    corners_reshaped = corners.reshape(3, -1)
                    proj = view_points(corners_reshaped, view=cam_intrinsic, normalize=True)[:2, :]
                    proj = proj.reshape(len(boxes), 2, 8)
                    mins = proj.min(axis=2)
                    maxs = proj.max(axis=2)

                    for min_vals, max_vals in zip(mins, maxs):
                        min_x, min_y = np.maximum(0, min_vals)
                        max_x, max_y = np.minimum([width, height], max_vals)

                        # Check if the bounding box is large enough (CLIP image encoder uses 224x224 inputs)
                        #CLIP_IMG_DIM = 224
                        CROP_TRESH = 50
                        if max_x - min_x >= CROP_TRESH and max_y - min_y >= CROP_TRESH:
                            # Collect image data for colorizing
                            cam_data = {}
                            cam_data['image_np'] = np.array(im)
                            cam_data['timestamp'] = cam_sd_rec['timestamp']
                            
                            cam_calib = dataset.get('calibrated_sensor', cam_sd_rec['calibrated_sensor_token'])
                            cam_pose = dataset.get('ego_pose', cam_sd_rec['ego_pose_token'])
                            T_cam = transform_matrix(cam_calib['translation'], Quaternion(cam_calib['rotation']), inverse=False)
                            T_ego = transform_matrix(cam_pose['translation'], Quaternion(cam_pose['rotation']), inverse=False)
                            T_camera = T_ego @ T_cam  # sensor -> global
                            T_global_to_cam = np.linalg.inv(T_camera)
                            
                            cam_data['T_global_to_cam'] = T_global_to_cam
                            cam_data['cam_intrinsic'] = cam_intrinsic
                            cam_data['ann_rec'] = ann_rec
                            
                            all_imgs_data.append(cam_data)
                            
                            # Crop image
                            cropped_im = im.crop((min_x, min_y, max_x, max_y))
                            
                            # Optional: visualize the cropped image
                            #plt.imshow(cropped_im)
                            #plt.show()

                            # Convert to numpy
                            cropped_im_np = np.array(cropped_im)

                            # Store it in our list
                            all_cropped_images.append((cropped_im_np, image_id_counter))
                            image_id_counter += 1
        """
        
        # LIDAR PROCESSING
        """
        #print(f"Number of pts in annotation: {ann_rec['num_lidar_pts']}")
        #if ann_rec['num_lidar_pts'] >= PTS_TRESH:
            
        #lidar_pts = load_lidar(dataset, sample_token)
            
        bbox = Box(center=ann_rec['translation'],
                size=ann_rec['size'],
                orientation=Quaternion(ann_rec['rotation']))
        in_mask = points_in_box(box=bbox, points=lidar_pts)
        
        if in_mask.sum() >= PTS_TRESH:
            cropped_lidar_pts = lidar_pts[:, in_mask]
            cropped_lidar_pts = cropped_lidar_pts.T # (N, 3)

            # Optionally visualize
            #print(f"Point cloud dim: {cropped_lidar_pts.shape}")
            #visualize_point_cloud(cropped_lidar_pts)

            #all_cropped_pcs.append((cropped_lidar_pts, pc_id_counter))
            all_cropped_pcs.append((cropped_lidar_pts, pc_id_counter))
            pc_id_counter += 1
        """
        
        # Use motion-compensated point clouds
        lidar_pts, point_times = load_and_fuse_lidar_motion_compensated(dataset, sample_token, ann_rec, max_sweeps=10, 
                                                                        point_cloud_cache=point_cloud_cache,
                                                                        sweep_cache=sweep_cache)
        
        # Crop at the end to avoid trails from incorrect velocity estimates
        bbox = Box(
            center=ann_rec['translation'],
            size=ann_rec['size'],
            orientation=Quaternion(ann_rec['rotation']))
        
        bbox_mask = points_in_box(box=bbox, points=lidar_pts)
        lidar_pts = lidar_pts[:, bbox_mask]

        # Continue with the existing cropping code
        if hasattr(lidar_pts, 'shape') and lidar_pts.shape[1] >= pts_tresh:
            cropped_lidar_pts = lidar_pts.T  # (N, 3)
            
            # Can optionally store time information
            #if point_times is not None:
            #    cropped_lidar_pts = np.column_stack((cropped_lidar_pts, point_times))
            
            all_cropped_pcs.append((cropped_lidar_pts, pc_id_counter))
            all_pcs_ann_rec.append(ann_rec)
            
            pc_id_counter += 1


    # Save data if >0 images and >0 point clouds
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist
    # if len(all_cropped_images) > 0 and len(all_cropped_pcs) > 0:
    if len(all_cropped_pcs) > 0:  # Only save point clouds
        
        instance_name = dataset.get('category', instance['category_token'])["name"]
        #print(f"Instance name: {instance_name}")
        
        """
        # Save the images
        images_dir = os.path.join(results_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        hdf5_path = os.path.join(images_dir, f"{instance_token}.hdf5")
        with h5py.File(hdf5_path, "w") as f:
            for img_array, img_id in all_cropped_images:
                dataset_name = f"image_{img_id}"
                # we can store it as a dataset
                f.create_dataset(dataset_name, data=img_array, compression="gzip")          

        print(f"Saved {len(all_cropped_images)} images for instance {instance_token} at {hdf5_path}")
        """
        
        # Save the point clouds
        pc_dir = os.path.join(results_dir, 'pc')
        os.makedirs(pc_dir, exist_ok=True)

        hdf5_path = os.path.join(pc_dir, f"{instance_token}.hdf5")
        with h5py.File(hdf5_path, "w") as f:
            for pc_array, pc_id in all_cropped_pcs:
                dset_name = f"pc_{pc_id}"
                
                """
                # Color the point cloud
                pc_array = colorize_point_cloud(
                                    pc_array, 
                                    all_imgs_data,
                                    pc_ann_rec=all_pcs_ann_rec[pc_id])       
                """
                
                # Center the points around their CoM
                center = np.mean(pc_array[:, :3], axis=0)
                pc_array[:, :3] -= center
                
                # Optionally visualize colored cloud
                #print(f"Point cloud dim: {pc_array.shape}")
                #visualize_point_cloud_o3d(pc_array)
                
                f.create_dataset(dset_name, data=pc_array, compression="gzip")
                
        # Save label (textual description)
        label = GENERAL_TO_DETECTION.get(instance_name, instance_name)
        
        labels_path = os.path.join(results_dir, "labels.txt")
        with open(labels_path, "a", encoding="utf-8") as f:
            f.write(f"{instance_token}:{label}\n")        
                
        #print(f"Saved {len(all_cropped_pcs)} point clouds for instance {instance_token} at {hdf5_path}")
    else:
        # save instance token to results/skipped.txt
        skipped_path = os.path.join(results_dir, 'skipped.txt')        
        with open(skipped_path, "a") as f:
            f.write(f"{instance_token}\n")
            
def get_gt_eval_boxes(dataset):
    from nuscenes.eval.common.loaders import (
        add_center_dist,
        filter_eval_boxes,
        load_gt)
    
    #from nuscenes.eval.common.data_classes import EvalBox
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.common.config import DetectionConfig
    
    gt_boxes = load_gt(dataset, "val", DetectionBox, verbose=True)
    gt_boxes = add_center_dist(dataset, gt_boxes)
    
    with open("detection_cvpr_2019.json", 'r') as f:
        cfg_data = json.load(f)
    cfg = DetectionConfig.deserialize(cfg_data)

    gt_boxes = filter_eval_boxes(dataset, gt_boxes, cfg.class_range, verbose=True)
    
    return gt_boxes


def process_eval_boxes(dataset, eval_boxes, results_dir, pts_tresh, point_cloud_cache=None, sweep_cache=None):
    
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
            # Get the original annotation record using the box token
            ann_rec = dataset.get('sample_annotation', box.ann_token)
            sample_token = box.sample_token
            
            # Use motion-compensated point clouds
            try:
                lidar_pts, point_times = load_and_fuse_lidar_motion_compensated(
                    dataset, sample_token, ann_rec, max_sweeps=10, 
                    point_cloud_cache=point_cloud_cache,
                    sweep_cache=sweep_cache)
                
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
    nusc = NuScenes(version=args.version,
                    dataroot=args.dataroot,
                    verbose=True)
    
    
    # if the dataset loaded is v1.0-trainval
    if args.version == 'v1.0-trainval':
        scene_splits = create_splits_scenes(verbose=True)
        scene_names = scene_splits[args.split]
    
    #labels_dict = {}
    results_dir = os.path.join(args.output, args.split if args.version == 'v1.0-trainval' else 'mini')
    
    point_cloud_cache = LRUCache(capacity=500)   # Cache for point clouds
    sweep_cache = LRUCache(capacity=2000)        # Cache for sweeps
    
    boxes = get_gt_eval_boxes(nusc)
    process_eval_boxes(nusc, boxes, results_dir, args.pts_tresh, point_cloud_cache, sweep_cache)
    
    print(f"\nFinished processing the specified split: {args.split}!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create nuScenes object triplets (3D and image cropping only).")
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='nuScenes dataset version')
    parser.add_argument('--split', type=str, default='val', help='Dataset split (train/val/test)')
    parser.add_argument('--dataroot', type=str, help='Path to nuScenes dataset root directory')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--pts_tresh', type=int, default=0, help='Minimum number of points in a bounding box to be considered valid')
    
    args = parser.parse_args()
    
    main(args)
