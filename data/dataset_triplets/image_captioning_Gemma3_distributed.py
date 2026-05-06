import os
import argparse
import json
import h5py
import torch
import torch.distributed as dist
from PIL import Image
from transformers import pipeline
from datasets import Dataset
from functools import partial
from huggingface_hub import login

login(token="TOKEN")  # Replace with your Hugging Face token

CLIP_TOKEN_LIMIT = 77

# System instructions.
# Immensely helps the model to get good results.
system_instructions = (
    "You are to caption images. "
    "Capture as much detail and semantic information as possible. "
    "Only describe one object, which is the largest one in the image. "
    "Ignore the background. "
    "Leave out image quality description from the caption."
)

# Gather image metadata without immediately loading
def load_image_mapping(images_dir):
    """
    Scan the 'images_dir' for all .hdf5 files and their respective datasets.
    Return a list of dicts, each describing one image, e.g.:
      {
        'hdf5_path': '...',
        'instance_id': '...',  # the filename minus .hdf5
        'image_key': 'image_0'
      }
    """
    image_mapping = []
    for fname in os.listdir(images_dir):
        if not fname.endswith(".hdf5"):
            continue
        hdf5_path = os.path.join(images_dir, fname)
        instance_id = fname[:-5]         # remove the trailing ".hdf5"
        
        with h5py.File(hdf5_path, "r") as f:
            image_keys = list(f.keys())  # e.g. ["image_0", "image_1", ...]
            
        for image_key in image_keys:
            image_mapping.append({
                "hdf5_path": hdf5_path,
                "instance_id": instance_id,
                "image_key": image_key
            })
    
    return image_mapping

# On-demand image loading
def load_images_in_batch(examples):
    """Load a batch of images from their HDF5 paths and image keys."""
    pil_images = []
    for hdf5_path, image_key in zip(examples["hdf5_path"], examples["image_key"]):
        with h5py.File(hdf5_path, "r") as f:
            img_data = f[image_key][()]
        pil_images.append(Image.fromarray(img_data))
    examples["image"] = pil_images
    return examples

# Generate captions in batches
def generate_captions_batch(examples, max_tokens=CLIP_TOKEN_LIMIT):
    """
    Given a batch of PIL images (examples["image"]), build the chat messages,
    pass them to the pipeline in a single call, and store the results in
    examples["caption"].
    """
    chat_batch = []
    for img in examples["image"]:
        single_chat = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_instructions}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": "A photo of"}
                ]
            }
        ]
        chat_batch.append(single_chat)

    # Call the pipeline once for the entire batch
    outputs = pipe(
        text=chat_batch,
        max_new_tokens=max_tokens
    )
    
    # Each element in 'outputs' corresponds to one item in the batch
    # We parse out the final generated caption
    captions = []
    for out in outputs:
        # out is typically a list of length 1 with a dict:
        # out[0]["generated_text"] is the list of chat messages from the model
        # The last entry has the actual content
        model_response = out[0]["generated_text"][-1]["content"]
        captions.append(model_response)

    examples["caption"] = captions
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to the dataset")
    parser.add_argument("--cache", help="Path to the Gemma 3 model cache directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for image loading and caption generation")
    
    args = parser.parse_args()
    
    images_dir = os.path.join(args.dataset, "images")
    
    # Adjust your cache and device settings
    if args.cache:
        cache_dir = args.cache
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = None
    
    # Set up distributed usage
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' and 'SLURM_NTASKS' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode. Exiting...')
        exit(0)
        
     # Initialize the distributed process group
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world_size,
                            rank=rank)
    
    torch.cuda.set_device(local_rank)
    torch.distributed.barrier()

    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-4b-it",
        device=f"cuda:{local_rank}",
        torch_dtype=torch.bfloat16,
        model_kwargs={"cache_dir": cache_dir}
    )

    # Generate image mapping and create dataset as before
    image_mapping = load_image_mapping(images_dir)
    ds = Dataset.from_list(image_mapping)

    # Shard the dataset for distributed processing
    #world_size = dist.get_world_size()
    #rank = dist.get_rank()
    ds = ds.shard(num_shards=world_size, index=rank)
    
    # Map functions to load images and generate captions
    ds = ds.map(load_images_in_batch,
                batched=True,
                batch_size=args.batch_size)
    
    ds = ds.map(partial(generate_captions_batch, max_tokens=CLIP_TOKEN_LIMIT),
                batched=True,
                batch_size=args.batch_size)
    
    # Collect results and save to a rank-specific file
    all_descriptions = {}
    for item in ds:
        inst_id = item["instance_id"]
        img_key = item["image_key"]
        caption = item["caption"]
        if inst_id not in all_descriptions:
            all_descriptions[inst_id] = {}
        all_descriptions[inst_id][img_key] = caption

    output_json_dir = os.path.join(args.dataset, f"captions_rank{rank}.json")
    with open(output_json_dir, "w", encoding="utf-8") as jf:
        json.dump(all_descriptions, jf, indent=4, ensure_ascii=False)

    print(f"Process {rank} has saved descriptions to {output_json_dir}")