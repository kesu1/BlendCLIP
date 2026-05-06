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

DIM_REDUCT_SEED = 42

def get_args_parser():
    parser = argparse.ArgumentParser(description='Visualize 3D PCA of model embeddings')
    
    # Add the required arguments from the original script
    parser.add_argument('--model', default='ULIP2_PointBERT', type=str, help='Model type')
    parser.add_argument('--test_ckpt_addr', required=True, help='Path to checkpoint')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers for dataloader')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use')
    parser.add_argument('--output-dir', default='./vis_outputs', type=str, help='Output directory')
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
    
    # Additional visualization args
    parser.add_argument('--subsample', type=float, default=0.0, help='Fraction of embedding points to subsample (0 for no subsampling, e.g., 0.1 for 10%)')
    parser.add_argument('--perplexity', type=float, default=30.0, help='Perplexity for t-SNE (if used)')
    parser.add_argument('--use-tsne', action='store_true', help='Use t-SNE instead of PCA')
    parser.add_argument('--no-class-color', action='store_true', help='Do not color by class, use single color for all points')
    parser.add_argument('--use-umap', action='store_true', help='Use UMAP instead of PCA') # New argument for UMAP
    parser.add_argument('--id', default='', type=str, help='Experiment ID for logging')
    
    return parser

def extract_embeddings_and_visualize(args):    
    # Try to load precomputed embeddings and labels if available
    output_base = os.path.join(args.output_dir, f"{args.validate_dataset_name}_{args.model.lower()}_{args.id}")

    embeddings_path = f"{output_base}_embeddings.npy"
    labels_path = f"{output_base}_labels.npy"
    label_to_name = None
    
    try:
        if 'objaverse' not in args.validate_dataset_name:
            with open(os.path.join("./data", 'labels.json')) as f:
                class_names = json.load(f)[args.validate_dataset_name]
                label_to_name = {i: name for i, name in enumerate(class_names)}
    except Exception as e:
        print(f"Could not load class names: {e}")
        return None

    if os.path.exists(embeddings_path) and os.path.exists(labels_path):
        print("Loading precomputed embeddings and labels...")
        all_embeddings = np.load(embeddings_path)
        all_labels = np.load(labels_path)
    else:
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
            multiprocessing_context=torch.multiprocessing.get_context("forkserver"),
            persistent_workers=True if args.workers > 0 else False
        )
        
        # Extract embeddings
        model.eval()
        all_embeddings = []
        all_labels = []
        all_names = []
        
        print("Extracting embeddings...")
        start_time = time.time()
        with torch.no_grad():
            for i, (pc, target, target_name) in enumerate(test_loader):
                # Handle dictionary-style point clouds
                if isinstance(pc, Mapping):  # pc: Sonata-style Point object
                    for key in pc.keys():
                        if isinstance(pc[key], torch.Tensor):
                            pc[key] = pc[key].cuda(args.gpu, non_blocking=True)
                else:
                    pc = pc.cuda(args.gpu, non_blocking=True)
                    
                # Encode point cloud
                pc_features = get_model(model).encode_pc(pc)
                pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
                
                # Store embeddings and labels
                all_embeddings.append(pc_features.cpu().numpy())
                all_labels.append(target.numpy())
                all_names.extend(target_name)
                
                if (i+1) % 10 == 0:
                    print(f"Processed {i+1}/{len(test_loader)} batches")
        
        # Concatenate all embeddings and labels
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
            
        print(f"Extracted {len(all_embeddings)} embeddings with dimension {all_embeddings.shape[1]}")
        print(f"Extraction took {time.time() - start_time:.2f} seconds")
        
        # Save embeddings and labels
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(embeddings_path, all_embeddings)
        np.save(labels_path, all_labels)
        
        """
        # Subsample if requested
    if args.subsample > 0.0 and args.subsample < 1.0:
        n_samples = int(len(all_embeddings) * args.subsample)
        print(f"Subsampling to {n_samples} embeddings ({args.subsample*100:.1f}%)")
        g = np.random.Generator(np.random.PCG64(42))
        indices = g.choice(len(all_embeddings), n_samples, replace=False)
        all_embeddings = all_embeddings[indices]
        all_labels = all_labels[indices]
        #all_names = [all_names[i] for i in indices]
      """
    """
    # Silhouette score calculation
    print("Calculating silhouette score...")
    start_time_sil = time.time()
    sil_embeddings = all_embeddings
    sil_labels = all_labels
    
    silhouette_avg = silhouette_score(sil_embeddings, sil_labels, metric='cosine')
    sil_time = time.time() - start_time_sil
    
    print(f"Silhouette Score (cosine similarity): {silhouette_avg:.4f}")
    print(f"Silhouette calculation took {sil_time:.2f} seconds")
    
    # Save silhouette score and details to text file
    silhouette_path = f"{output_base}_silhouette_details.txt"
    with open(silhouette_path, 'w') as f:
        f.write(f"Silhouette Score Analysis\n")
        f.write(f"========================\n\n")
        f.write(f"Dataset: {args.validate_dataset_name}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Experiment ID: {args.id}\n")
        f.write(f"Total embeddings: {len(all_embeddings)}\n")
        f.write(f"Embedding dimension: {all_embeddings.shape[1]}\n")
        f.write(f"Number of classes: {len(np.unique(all_labels))}\n")
        f.write(f"Samples used for silhouette: {len(sil_embeddings)}\n")
        f.write(f"Similarity metric: cosine\n")
        f.write(f"Silhouette Score: {silhouette_avg:.6f}\n")
        f.write(f"Calculation time: {sil_time:.2f} seconds\n\n")
        f.write(f"Class distribution:\n")
    
    print(f"Silhouette details saved to {silhouette_path}")    
    """
    
    # Perform dimensionality reduction
    print("Performing dimensionality reduction...")
    start_time = time.time()
        
    if args.use_tsne:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=args.perplexity, max_iter=5000, random_state=DIM_REDUCT_SEED)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method_name = 't-SNE'
    elif args.use_umap:
        try:
            import umap
        except ImportError:
            print("UMAP not installed. Falling back to PCA.")
            return None
        reducer = umap.UMAP(n_components=2, random_state=DIM_REDUCT_SEED)
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        method_name = 'UMAP'
    else:
        # PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(all_embeddings)
        explained_variance = pca.explained_variance_ratio_
        total_variance = sum(explained_variance)
        print(f"Explained variance ratio: {explained_variance}")
        print(f"Total variance explained: {total_variance:.4f}")
        method_name = 'PCA'
    
    print(f"Dimensionality reduction took {time.time() - start_time:.2f} seconds")
    
            # Subsample if requested
    if args.subsample > 0.0 and args.subsample < 1.0:
        n_samples = int(len(reduced_embeddings) * args.subsample)
        print(f"Subsampling to {n_samples} embeddings ({args.subsample*100:.1f}%)")
        g = np.random.Generator(np.random.PCG64(42))
        indices = g.choice(len(reduced_embeddings), n_samples, replace=False)
        reduced_embeddings = reduced_embeddings[indices]
        all_labels = all_labels[indices]
        #all_names = [all_names[i] for i in indices]
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Dim1': reduced_embeddings[:, 0],
        'Dim2': reduced_embeddings[:, 1],
        'label': all_labels
    })
        
    # Convert numeric labels to class names if available
    #if label_to_name:
    #    df['class_name'] = df['label'].map(label_to_name)
    
    # Create 3D scatter plot
    print("Creating visualization...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    
     # Set axis limits to fit the data tightly
    ax.set_xlim(df['Dim1'].min(), df['Dim1'].max())
    ax.set_ylim(df['Dim2'].min(), df['Dim2'].max())
    
    if args.no_class_color:
        # All points same color
        norm = plt.Normalize(df['Dim1'].min(), df['Dim1'].max())
        colors = plt.cm.viridis(norm(df['Dim1']))
        ax.scatter(
            df['Dim1'], df['Dim2'],
            s=0.1,
            c=colors,
            alpha=0.6,
            marker=',',
        )
    else:
        # Class-wise coloring
        unique_labels = sorted(df['label'].unique())
        n_classes = len(unique_labels)
        #colors = sns.color_palette('hsv', n_classes)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes)) #for 10 classes
        for i, label in enumerate(unique_labels):
            idx = df['label'] == label
            label_name = label_to_name[label] if label_to_name else f"Class {label}"
            ax.scatter(
                df.loc[idx, 'Dim1'],
                df.loc[idx, 'Dim2'],
                s=50,  # Reduced point size
                color=colors[i],
                alpha=1,
                label=label_name,
                marker="o"
            )
            print(f"{to_hex(colors[i])}: {label_name}")
        
    # Remove legend and title
    ax.legend_.remove() if hasattr(ax, 'legend_') and ax.legend_ else None
    ax.set_title('')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"{output_base}_{method_name}_vis.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Plot saved to {output_base}_{method_name}_vis.png")
    
    print("Visualization complete!")
    
    return {
        'embeddings': all_embeddings,
        'labels': all_labels, 
        'reduced_embeddings': reduced_embeddings
    }

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    extract_embeddings_and_visualize(args)

if __name__ == "__main__":
    main()