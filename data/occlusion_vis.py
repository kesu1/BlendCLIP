#!/usr/bin/env python3
"""
Objaverse Point Cloud Visualization Script
Loads an Objaverse sample and visualizes it with and without occlusion simulation.
"""

import os
import sys
import numpy as np
import open3d as o3d
import random
import argparse
import time

# Add the parent directory to the path to import from dataset_3d.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from dataset_3d.py
from data.dataset_3d import simulate_occlusion, pc_normalize


def load_objaverse_sample(data_path, sample_id=None):
    """
    Load an Objaverse point cloud sample.
    
    Args:
        data_path: Path to objaverse_pc_parallel directory
        sample_id: Specific sample ID to load, or None for random
    
    Returns:
        point_cloud: numpy array of shape (N, 6) with XYZ+RGB
        sample_name: name of the loaded sample
    """
    # Get all .npz files
    npz_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
    
    if not npz_files:
        raise ValueError(f"No .npz files found in {data_path}")
    
    # Select sample
    if sample_id is None:
        selected_file = random.choice(npz_files)
        print(f"Randomly selected: {selected_file}")
    else:
        # Look for file starting with sample_id
        matching_files = [f for f in npz_files if f.startswith(sample_id)]
        if not matching_files:
            raise ValueError(f"No files found starting with {sample_id}")
        selected_file = matching_files[0]
        print(f"Selected: {selected_file}")
    
    # Load the point cloud
    file_path = os.path.join(data_path, selected_file)
    try:
        with np.load(file_path) as data:
            point_cloud = data['point_cloud']
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")
    
    # Normalize coordinates using the function from dataset_3d.py
    point_cloud[:, 0:3] = pc_normalize(point_cloud[:, 0:3])
    
    sample_name = selected_file.split('_')[0]
    print(f"Loaded point cloud with {point_cloud.shape[0]} points")
    print(f"Point cloud shape: {point_cloud.shape}")
    
    return point_cloud, sample_name


def create_point_cloud_o3d(points, colors=None):
    """Create Open3D point cloud object."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    if colors is not None:
        # Ensure colors are in [0,1] range
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif points.shape[1] >= 6:
        # Use RGB from point cloud
        colors = points[:, 3:6]
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def create_camera_marker(camera_pos, size=0.1):
    """Create a visual marker for camera position."""
    camera_marker = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    camera_marker.translate(camera_pos)
    camera_marker.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
    return camera_marker


def visualize_dual_windows(original_points, occluded_points, camera_pos, sample_name):
    """Visualize original and occluded point clouds in separate windows with same camera angle."""
    
    # Create point clouds
    pcd_original = create_point_cloud_o3d(original_points)
    pcd_occluded = create_point_cloud_o3d(occluded_points)
    
    # Create camera markers
    camera_marker1 = create_camera_marker(camera_pos, size=0.05)
    camera_marker2 = create_camera_marker(camera_pos, size=0.05)
    
    # Create coordinate frames
    #coord_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    #coord_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    
    # Print info
    print(f"\nVisualization Info:")
    print(f"Sample: {sample_name}")
    print(f"Original points: {len(original_points)}")
    print(f"Occluded points: {len(occluded_points)}")
    print(f"Occlusion ratio: {1 - len(occluded_points)/len(original_points):.2%}")
    print(f"Camera position: {camera_pos}")
    print(f"\nTwo separate windows will open:")
    print(f"First window: Original point cloud")
    print(f"Second window: Occluded point cloud")
    
    # Create visualizers
    vis1 = o3d.visualization.Visualizer()
    vis2 = o3d.visualization.Visualizer()
    
    # Create windows
    vis1.create_window(window_name=f"Original - {sample_name}", width=800, height=600, left=100, top=100)
    vis2.create_window(window_name=f"Occluded - {sample_name}", width=800, height=600, left=950, top=100)
    
    # Add geometries to first window (original)
    vis1.add_geometry(pcd_original)
    vis1.add_geometry(camera_marker1)
    #vis1.add_geometry(coord_frame1)
    
    # Add geometries to second window (occluded)
    vis2.add_geometry(pcd_occluded)
    vis2.add_geometry(camera_marker2)
    #vis2.add_geometry(coord_frame2)
    
    # Set the same view for both windows
    # Get view control for both visualizers
    view_ctrl1 = vis1.get_view_control()
    view_ctrl2 = vis2.get_view_control()
    
    # Set a good default view
    view_ctrl1.set_front([0.3, 0.3, 0.9])
    view_ctrl1.set_up([0, 1, 0])
    view_ctrl1.set_lookat([0, 0, 0])
    view_ctrl1.set_zoom(0.8)
    
    # Copy the same view to the second window
    view_ctrl2.set_front([0.3, 0.3, 0.9])
    view_ctrl2.set_up([0, 1, 0])
    view_ctrl2.set_lookat([0, 0, 0])
    view_ctrl2.set_zoom(0.8)
    
    # Run the visualization loop
    try:
        while True:
            # Update both visualizers
            if not vis1.poll_events():
                break
            if not vis2.poll_events():
                break
                
            vis1.update_renderer()
            vis2.update_renderer()
            
            time.sleep(0.01)  # Small delay to prevent high CPU usage
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    finally:
        # Clean up
        vis1.destroy_window()
        vis2.destroy_window()


def visualize_single_window_comparison(original_points, occluded_points, camera_pos, sample_name):
    """Alternative: Visualize original and occluded point clouds side by side in single window."""
    
    # Create point clouds
    pcd_original = create_point_cloud_o3d(original_points)
    pcd_occluded = create_point_cloud_o3d(occluded_points)
    
    # Offset occluded point cloud for side-by-side view
    offset = np.array([3.0, 0.0, 0.0])
    pcd_occluded.translate(offset)
    
    # Create camera markers
    camera_marker1 = create_camera_marker(camera_pos, size=0.05)
    camera_marker2 = create_camera_marker(camera_pos + offset, size=0.05)
    
    # Create coordinate frames
    coord_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    coord_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    coord_frame2.translate(offset)
    
    # Print info
    print(f"\nVisualization Info:")
    print(f"Sample: {sample_name}")
    print(f"Original points: {len(original_points)}")
    print(f"Occluded points: {len(occluded_points)}")
    print(f"Occlusion ratio: {1 - len(occluded_points)/len(original_points):.2%}")
    print(f"Camera position: {camera_pos}")
    print(f"\nLeft: Original | Right: Occluded")
    
    # Visualize
    geometries = [pcd_original, pcd_occluded, camera_marker1, camera_marker2, 
                  coord_frame1, coord_frame2]
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Objaverse Sample: {sample_name} - Original vs Occluded",
        width=1200,
        height=800
    )


def main():
    parser = argparse.ArgumentParser(description='Visualize Objaverse point clouds with occlusion simulation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to objaverse_pc_parallel directory')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Specific sample ID to load (e.g., "000074a334c541878360457c672b6c2e")')
    parser.add_argument('--occlusion_param', type=float, default=2.0,
                        help='Occlusion parameter (default: 2.0)')
    parser.add_argument('--max_points', type=int, default=10000,
                        help='Maximum number of points to visualize (for performance)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--single_window', action='store_true',
                        help='Use single window side-by-side view instead of dual windows')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Validate data path
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        return
    
    try:
        # Load sample
        print(f"Loading Objaverse sample from {args.data_path}")
        original_points, sample_name = load_objaverse_sample(args.data_path, args.sample_id)
        
        # Subsample if too many points
        if original_points.shape[0] > args.max_points:
            indices = np.random.choice(original_points.shape[0], args.max_points, replace=False)
            original_points = original_points[indices, :]
            print(f"Subsampled to {args.max_points} points for visualization")
        
        # Apply occlusion simulation using the function from dataset_3d.py
        print(f"Applying occlusion simulation with parameter {args.occlusion_param}")
        occluded_points, camera_pos = simulate_occlusion(original_points, param=args.occlusion_param)
        
        # Ensure minimum points after occlusion
        attempt = 0
        while occluded_points.shape[0] < 500 and attempt < 10:
            print(f"Too few points after occlusion ({occluded_points.shape[0]}), retrying...")
            occluded_points, camera_pos = simulate_occlusion(original_points, param=args.occlusion_param)
            attempt += 1
        
        if occluded_points.shape[0] < 500:
            print("Warning: Still few points after multiple attempts, proceeding anyway...")
        
        # Visualize comparison
        if args.single_window:
            visualize_single_window_comparison(original_points, occluded_points, camera_pos, sample_name)
        else:
            visualize_dual_windows(original_points, occluded_points, camera_pos, sample_name)
        
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()