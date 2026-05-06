import argparse
import os
import torch
import numpy as np
import h5py
from PIL import Image
from collections import OrderedDict
import json

# Import from ULIP project
import models.ULIP_models as models
from utils.utils import get_model

def get_args_parser():
    parser = argparse.ArgumentParser(description='3D Retrieval from precomputed embeddings')
    
    # Retrieval Specific Arguments
    parser.add_argument('--prompt', required=True, type=str, help='Text prompt for retrieval (e.g., "ambulance").')
    parser.add_argument('--k', default=5, type=int, help='Number of top results to retrieve.')
    parser.add_argument('--embeddings_path', required=True, type=str, help='Path to precomputed embeddings .npy file.')
    parser.add_argument('--img_paths_path', required=True, type=str, help='Path to image paths .npy file.')
    parser.add_argument('--retrieval_output_dir', default='./retrieval_results', type=str, help='Directory to save retrieved images.')
    parser.add_argument('--templates_key', default='outdoors_1', type=str, help='Key for templates in templates.json.')

    # Not Retrieval Specific Arguments
    parser.add_argument('--model', default='ULIP2_PointBERT', type=str, help='Model type')
    parser.add_argument('--test_ckpt_addr', required=True, help='Path to checkpoint')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use')
    parser.add_argument('--linear-projection', action='store_true', help='use linear projection instead of MLP')
    #parser.add_argument('--pooling-type', default='mean', type=str, help='pooling type', choices=['sum', 'mean', 'mix', 'max'])
    parser.add_argument('--evaluate_3d', action='store_true', help='Evaluate 3D zero-shot')

    
    return parser

def retrieve_top_k(args):
    """
    Computes similarity, finds top-k matches, and saves the corresponding images.
    """
    print("Starting retrieval process...")
    
    # Setup
    os.makedirs(args.retrieval_output_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Load Precomputed Data
    print(f"Loading embeddings from {args.embeddings_path}")
    try:
        pc_embeddings = np.load(args.embeddings_path)
        img_paths = np.load(args.img_paths_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the embedding and path files exist.")
        return

    print(f"Loaded {len(pc_embeddings)} point cloud embeddings.")
    pc_embeddings = torch.from_numpy(pc_embeddings).cuda(args.gpu)

    # Load Model for Text Encoding
    print(f"=> creating model: {args.model}")
    model, tokenizer, _, _ = getattr(models, args.model)(args=args)
    
    # Load checkpoint
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    model.cuda(args.gpu)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"=> loaded pretrained checkpoint '{args.test_ckpt_addr}'")

    # Encode Text Prompt using Templates
    print(f"Encoding prompt: '{args.prompt}' using templates from '{args.templates_key}'")
    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.templates_key]

    with torch.no_grad():
        texts = [t.format(args.prompt) for t in templates]
        tokenized_texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
        
        class_embeddings = get_model(model).encode_text(tokenized_texts)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features = class_embeddings.mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.unsqueeze(0) # Keep shape [1, D]

    # Compute Similarity and Find Top-K
    print("Calculating similarities...")
    similarities = text_features @ pc_embeddings.T
    similarities = similarities.squeeze(0).cpu().numpy()

    # Get top-k indices
    top_k_indices = np.argsort(similarities)[-args.k:][::-1]
    top_k_scores = similarities[top_k_indices]
    top_k_paths = img_paths[top_k_indices]

    print(f"\nTop {args.k} results for prompt: '{args.prompt}'")

    # Save Top-K Results
    for i, (idx, score, path) in enumerate(zip(top_k_indices, top_k_scores, top_k_paths)):
        print(f"  {i+1}. Score: {score:.4f}, Path: {path}")
        
        try:
            with h5py.File(path, 'r') as hf:
                img_keys = list(hf.keys())
                if not img_keys:
                    print(f"Warning: No images found in {path}. Skipping.")
                    continue
                
                instance_id = os.path.basename(path).split('.')[0]

                for img_idx, key in enumerate(img_keys):
                    img_data = hf[key][:]
                    
                    # Convert to PIL Image
                    if len(img_data.shape) == 3 and img_data.shape[2] == 3:  # HWC format
                        img = Image.fromarray(img_data)
                    elif len(img_data.shape) == 3 and img_data.shape[0] == 3:  # CHW format
                        img = Image.fromarray(np.transpose(img_data, (1, 2, 0)))
                    else:
                        print(f"Warning: Unexpected image shape {img_data.shape} for key {key} in {path}. Skipping.")
                        continue

                    output_filename = f"{instance_id}_retrieval{i+1}_img{img_idx}.png"
                    output_path = os.path.join(args.retrieval_output_dir, output_filename)
                    img.save(output_path)
                    print(f"Saved image to {output_path}")

        except Exception as e:
            print(f"Error processing file {path}: {e}")

    print(f"\nRetrieval complete. Results saved in {args.retrieval_output_dir}")


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    retrieve_top_k(args)

if __name__ == "__main__":
    main()