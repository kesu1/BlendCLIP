#!/usr/bin/env python3
"""
Script to visualize a randomly selected triplet (point cloud, image, text) from the dataset.
Supports both Objaverse and NuScenes data.
"""

from email import parser
import os
import random
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
import argparse


def pc_normalize(pc):
    """Normalize point cloud to unit sphere"""
    if pc.shape[0] == 0:
        return pc
    
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    
    if m == 0:
        return pc
    
    pc = pc / m
    return pc


def load_nuscenes_triplet(nuscenes_root, args):
    """Load a random triplet from NuScenes data"""
    print("Loading from NuScenes dataset...")
    
    # Paths
    pc_dir = os.path.join(nuscenes_root, "train", "pc")
    img_dir = os.path.join(nuscenes_root, "train", "images")
    caption_file = os.path.join(nuscenes_root, "train", "captions.json")
    
    # Check if paths exist
    if not os.path.exists(pc_dir):
        raise FileNotFoundError(f"NuScenes PC directory not found: {pc_dir}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"NuScenes images directory not found: {img_dir}")
    if not os.path.exists(caption_file):
        raise FileNotFoundError(f"NuScenes captions file not found: {caption_file}")
    
    # Load captions
    with open(caption_file, 'r') as f:
        captions = json.load(f)
    
    # Get all point cloud files
    pc_files = [f for f in os.listdir(pc_dir) if f.endswith(".hdf5")]
    if not pc_files:
        raise FileNotFoundError("No HDF5 point cloud files found in NuScenes PC directory")
    
    if not args.instance:
        # Randomly select a file
        selected_file = random.choice(pc_files)
        instance_id = selected_file[:-5]  # remove .hdf5
    else:
        # Use specified instance ID
        instance_id = args.instance
        selected_file = f"{instance_id}.hdf5"
        if selected_file not in pc_files:
            raise FileNotFoundError(f"Specified instance {instance_id} not found in PC directory")
        
    
    print(f"Selected NuScenes instance: {instance_id}")
    
    # Load point cloud
    pc_path = os.path.join(pc_dir, selected_file)
    with h5py.File(pc_path, 'r') as f:
        pc_keys = list(f.keys())
        if not pc_keys:
            raise ValueError(f"No keys found in point cloud file: {pc_path}")
        
        # Randomly select a key
        selected_key = random.choice(pc_keys)
        point_cloud = f[selected_key][:]
        
        print(f"Selected PC key: {selected_key}")
    
    # Normalize point cloud
    if point_cloud.shape[1] < 3:
        point_cloud = point_cloud.transpose(0, 2, 1)
    
    point_cloud[:, 0:3] = pc_normalize(point_cloud[:, 0:3])
    
    # Load image
    img_path = os.path.join(img_dir, selected_file)
    with h5py.File(img_path, 'r') as f:
        img_keys = list(f.keys())
        if not img_keys:
            raise ValueError(f"No keys found in image file: {img_path}")
        
        # Randomly select an image key
        selected_img_key = random.choice(img_keys)
        img_data = f[selected_img_key][:]
        
        print(f"Selected image key: {selected_img_key}")
    
    # Convert image to PIL
    if len(img_data.shape) == 3:
        if img_data.shape[0] <= 4 and img_data.shape[0] < img_data.shape[1]:
            # CHW format
            img_data = np.transpose(img_data, (1, 2, 0))
        
        if img_data.shape[2] == 3:  # RGB
            image = Image.fromarray(img_data)
        elif img_data.shape[2] == 4:  # RGBA
            image = Image.fromarray(img_data).convert('RGB')
        elif img_data.shape[2] == 1:  # Grayscale
            image = Image.fromarray(img_data.squeeze()).convert('RGB')
        else:
            raise ValueError(f"Unsupported number of channels: {img_data.shape[2]}")
    else:
        raise ValueError(f"Unexpected image dimensions: {img_data.shape}")
    
    # Get caption
    if instance_id in captions and selected_img_key in captions[instance_id]:
        caption = captions[instance_id][selected_img_key]
    else:
        caption = f"No caption found for {instance_id}/{selected_img_key}"
    
    return point_cloud, image, caption, f"nuscenes_{instance_id}_{selected_key}_{selected_img_key}"


def load_objaverse_triplet(objaverse_root, args):
    """Load a random triplet from Objaverse data"""
    print("Loading from Objaverse dataset...")
    
    # Paths
    pc_dir = os.path.join(objaverse_root, "objaverse_pc_parallel")
    img_dir = os.path.join(objaverse_root, "rendered_images_split_100")
    caption_file = os.path.join(objaverse_root, "merged_data.json")
    
    # Check if paths exist
    if not os.path.exists(pc_dir):
        raise FileNotFoundError(f"Objaverse PC directory not found: {pc_dir}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Objaverse images directory not found: {img_dir}")
    if not os.path.exists(caption_file):
        raise FileNotFoundError(f"Objaverse captions file not found: {caption_file}")
    
    # Load captions
    with open(caption_file, 'r') as f:
        captions = json.load(f)
    
    # Get all point cloud files
    pc_files = [f for f in os.listdir(pc_dir) if f.endswith(".npz")]
    if not pc_files:
        raise FileNotFoundError("No NPZ point cloud files found in Objaverse PC directory")
    
    if not args.instance:
        # Randomly select a file
        selected_file = random.choice(pc_files)
        instance_id = selected_file.split('_')[0]
    else:
        # Use specified instance ID
        instance_id = args.instance
        selected_file = f"{instance_id}._10000.hdf5"
        if selected_file not in pc_files:
            raise FileNotFoundError(f"Specified instance {instance_id} not found in PC directory")

    print(f"Selected Objaverse instance: {instance_id}")
    
    # Load point cloud
    pc_path = os.path.join(pc_dir, selected_file)
    with np.load(pc_path) as data:
        point_cloud = data['point_cloud']
    
    # Normalize point cloud
    point_cloud[:, 0:3] = pc_normalize(point_cloud[:, 0:3])
    
    # Load image
    img_path = os.path.join(img_dir, f"{instance_id}.hdf5")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    with h5py.File(img_path, 'r') as f:
        img_keys = list(f.keys())
        if not img_keys:
            raise ValueError(f"No keys found in image file: {img_path}")
        
        # Randomly select an image key
        selected_img_key = random.choice(img_keys)
        img_data = f[selected_img_key][:]
        
        print(f"Selected image key: {selected_img_key}")
    
    # Convert image to PIL
    if len(img_data.shape) == 3:
        if img_data.shape[0] <= 4 and img_data.shape[0] < img_data.shape[1]:
            # CHW format
            img_data = np.transpose(img_data, (1, 2, 0))
        
        if img_data.shape[2] == 3:  # RGB
            image = Image.fromarray(img_data)
        elif img_data.shape[2] == 4:  # RGBA
            image = Image.fromarray(img_data).convert('RGB')
        elif img_data.shape[2] == 1:  # Grayscale
            image = Image.fromarray(img_data.squeeze()).convert('RGB')
        else:
            raise ValueError(f"Unsupported number of channels: {img_data.shape[2]}")
    else:
        raise ValueError(f"Unexpected image dimensions: {img_data.shape}")
    
    # Get caption
    caption_key = f"/export/einstein-vision/3d_vision/objaverse/render_images_split_100/{instance_id}/{selected_img_key}"
    if caption_key in captions:
        # Use the first caption (as mentioned in the request)
        caption = captions[caption_key][0] if isinstance(captions[caption_key], list) else captions[caption_key]
    else:
        caption = f"No caption found for {caption_key}"
    
    return point_cloud, image, caption, f"objaverse_{instance_id}_{selected_img_key}"


def visualize_triplet(point_cloud, image, caption, sample_id):
    """Visualize the triplet: point cloud with Open3D, save image and caption to files"""
    
    print(f"\n{'='*50}")
    print(f"Sample ID: {sample_id}")
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Image size: {image.size}")
    print(f"Caption: {caption}")
    print(f"{'='*50}\n")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    
    # Add colors if available
    #if point_cloud.shape[1] >= 6:
    #    colors = point_cloud[:, 3:6]
    #    # Normalize colors to [0, 1] if they're in [0, 255] range
    #    if colors.max() > 1.0:
    #        colors = colors / 255.0
    #    pcd.colors = o3d.utility.Vector3dVector(colors)
    #else:
    # Set uniform color if no color data
    colors = np.tile([0.7, 0.7, 0.7], (len(point_cloud), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save image and caption to files
    img_filename = f"{sample_id}.png"
    caption_filename = f"{sample_id}.txt"
    
    # Save image
    image.save(img_filename)
    print(f"Image saved to: {img_filename}")
    
    # Save caption
    with open(caption_filename, 'w', encoding='utf-8') as f:
        f.write(caption)
    print(f"Caption saved to: {caption_filename}")
    
    # Visualize point cloud with Open3D
    print("Displaying point cloud with Open3D...")
    #o3d.visualization.draw_geometries([pcd], 
    #                                window_name=f"Point Cloud: {sample_id}",
    #                                width=800, 
    #                                height=600)
        # Create visualization with custom render options
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud: {sample_id}", width=800, height=600)
    vis.add_geometry(pcd)
    
    # Get render option and modify point visualization
    render_option = vis.get_render_option()
    render_option.point_size = 8.0  # Make points bigger (default is usually 1.0)
    render_option.point_show_normal = False
    render_option.point_color_option = o3d.visualization.PointColorOption.Default
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()



def main():
    parser = argparse.ArgumentParser(description="Visualize a random triplet from the dataset")
    parser.add_argument("--dataset", choices=["objaverse", "nuscenes", "random"], 
                       default="random", 
                       help="Choose dataset to load from (default: random)")
    parser.add_argument("--nuscenes_path", type=str, 
                       default="/proj/berzelius-2023-364/data/nuscenes_objects",
                       help="Path to NuScenes data")
    parser.add_argument("--objaverse_path", type=str, 
                       default="/proj/berzelius-2023-364/data/Objaverse_triplets",
                       help="Path to Objaverse data")
    parser.add_argument("--instance", type=str, 
                        default="",
                        help="Instance ID to visualize (if not set, a random instance is shown)")
    
    
    args = parser.parse_args()
    
    # Randomly choose dataset if not specified
    if args.dataset == "random":
        # Check which datasets are available
        available_datasets = []
        if os.path.exists(args.nuscenes_path):
            available_datasets.append("nuscenes")
        if os.path.exists(args.objaverse_path):
            available_datasets.append("objaverse")
        
        if not available_datasets:
            print("No datasets found. Please check the paths:")
            print(f"NuScenes: {args.nuscenes_path}")
            print(f"Objaverse: {args.objaverse_path}")
            return
        
        chosen_dataset = random.choice(available_datasets)
        print(f"Randomly selected dataset: {chosen_dataset}")
    else:
        chosen_dataset = args.dataset
    
    try:
        if chosen_dataset == "nuscenes":
            point_cloud, image, caption, sample_id = load_nuscenes_triplet(args.nuscenes_path, args)
        elif chosen_dataset == "objaverse":
            point_cloud, image, caption, sample_id = load_objaverse_triplet(args.objaverse_path, args)
        else:
            raise ValueError(f"Unknown dataset: {chosen_dataset}")
        
        visualize_triplet(point_cloud, image, caption, sample_id)
        
    except Exception as e:
        print(f"Error loading triplet: {str(e)}")
        print("Please check that the data paths are correct and contain the expected structure.")


if __name__ == "__main__":
    main()
