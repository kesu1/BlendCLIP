import argparse
import os
from matplotlib import colors
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from typing import Mapping
import json
from collections import OrderedDict
import time
from sklearn.metrics import silhouette_score
from matplotlib.colors import to_hex

# Import from ULIP project
import models.ULIP_models as models
from utils.utils import get_dataset, get_model
from data.dataset_3d import customized_collate_fn
from utils.utils import init_distributed_mode

def get_args_parser():
    parser = argparse.ArgumentParser(description='Visualize 3D PCA of model embeddings')
    
    # Add the required arguments from the original script
    parser.add_argument('--model', default='ULIP2_PointBERT', type=str, help='Model type')
    parser.add_argument('--test_ckpt_addr', required=True, help='Path to checkpoint')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers for dataloader')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use')
    parser.add_argument('--output-dir', default='./precomputed', type=str, help='Output directory')
    parser.add_argument('--validate_dataset_name', required=True, type=str, help='Dataset name')
    parser.add_argument('--validate_dataset_prompt', required=True, type=str, help='Dataset prompt')
    parser.add_argument('--npoints', default=8192, type=int, help='Number of points used for test.')
    parser.add_argument('--use_height', action='store_true', help='Whether to use height information')
    parser.add_argument('--evaluate_3d', action='store_true', help='Evaluate 3D zero-shot')
    parser.add_argument('--pretrain_dataset_name', default='objaverse', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--sim-occlusion', action='store_true', help='use occlusion simulation')
    parser.add_argument('--linear-projection', action='store_true', help='use linear projection instead of MLP')
    #parser.add_argument('--pooling-type', default='mean', type=str, help='pooling type', choices=['sum', 'mean', 'mix', 'max'])
    
     # Args for distributed training
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--no-distributed', action='store_true', help='disable distributed training completely')
    parser.add_argument('--excluded-classes', nargs='*', default=[], type=str, help='space separated list of classes to exclude from training (for nuscenes_objects)')

    
    # Additional visualization args
    parser.add_argument('--id', default='', type=str, help='Experiment ID for logging')
    
    return parser

def extract_embeddings(args):    
    # Try to load precomputed embeddings and labels if available
    output_base = os.path.join(args.output_dir, f"{args.validate_dataset_name}_{args.model.lower()}_{args.id}")

    embeddings_path = f"{output_base}_embeddings.npy"
    img_paths_path = f"{output_base}_img_paths.npy"
    
    """
    try:
        if 'objaverse' not in args.validate_dataset_name:
            with open(os.path.join("./data", 'labels.json')) as f:
                class_names = json.load(f)[args.validate_dataset_name]
                label_to_name = {i: name for i, name in enumerate(class_names)}
    except Exception as e:
        print(f"Could not load class names: {e}")
        return None
    """
    
    # Initialize distributed mode if needed
    init_distributed_mode(args)
        
    # Clear CUDA cache before model and data loading if extracting embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Load checkpoint
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print(f"=> creating model: {args.model}")

    # Create model
    model, tokenizer, _, _ = getattr(models, args.model)(args=args)
    model.cuda()
    model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded pretrained checkpoint '{args.test_ckpt_addr}'")
        
    # Get dataset and loader
    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False,
        collate_fn=customized_collate_fn,
        #multiprocessing_context=torch.multiprocessing.get_context("forkserver"),
        persistent_workers=True if args.workers > 0 else False
    )
        
    # Extract embeddings
    model.eval()
    all_embeddings = []
    all_img_paths = []
        
    print("Extracting embeddings...")
    with torch.no_grad():
        for i, (pc, target, img_path) in enumerate(test_loader):
            
            pc = pc.cuda(args.gpu, non_blocking=True)
                    
            # Encode point cloud
            pc_features = get_model(model).encode_pc(pc)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
                
            # Store embeddings and labels
            all_embeddings.append(pc_features.cpu().numpy())
            all_img_paths.extend(img_path)
                
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(test_loader)} batches")
        
    # Concatenate all embeddings and labels
    all_embeddings = np.vstack(all_embeddings)

    print(f"Extracted {len(all_embeddings)} embeddings with dimension {all_embeddings.shape[1]}")
        
    # Save embeddings and labels
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(embeddings_path, all_embeddings)
    np.save(img_paths_path, np.array(all_img_paths))
    
    print("Finished saving embeddings and labels!")
    
def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    extract_embeddings(args)

if __name__ == "__main__":
    main()