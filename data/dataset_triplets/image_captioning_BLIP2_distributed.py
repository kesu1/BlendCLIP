import os
import argparse
import json
import h5py
import torch
import torch.distributed as dist
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from datasets import Dataset
from functools import partial

CLIP_TOKEN_LIMIT = 77

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
def generate_captions_batch(examples, processor, model, device, max_tokens=CLIP_TOKEN_LIMIT):
    """
    Given a batch of PIL images (examples["image"]), generate captions using BLIP2.
    """
    PROMPT = "a photo of"
    
    # Prepare batch inputs
    prompts = [PROMPT] * len(examples["image"])

    # Process batch of images and prompts
    inputs = processor(
        images=examples["image"], 
        text=prompts, 
        return_tensors="pt"
        #padding=True
    ).to(device)
    
    # Generate captions for the batch
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )
    
    # Decode the generated captions
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    captions = [caption.replace(PROMPT, "").strip() for caption in captions]
    
    examples["caption"] = captions
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to the dataset")
    parser.add_argument("--cache", help="Path to the BLIP2 model cache directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for image loading and caption generation")
    
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
        local_rank = rank % torch.cuda.device_count()
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

    # Initialize BLIP2 model and processor
    device = f"cuda:{local_rank}"
    
    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-6.7b",
        cache_dir=cache_dir,
    )
    
    """
    # Configure 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    """

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-6.7b",
        cache_dir=cache_dir,
        #torch_dtype=torch.bfloat16,
        #quantization_config=bnb_config,
        device_map={"": local_rank}
    )

    # Generate image mapping and create dataset
    image_mapping = load_image_mapping(images_dir)
    ds = Dataset.from_list(image_mapping)

    # Shard the dataset for distributed processing
    ds = ds.shard(num_shards=world_size, index=rank)
    
    # Map functions to load images
    ds = ds.map(load_images_in_batch,
                batched=True,
                batch_size=args.batch_size)
    
    # Map function to generate captions
    ds = ds.map(partial(generate_captions_batch, 
                       processor=processor, 
                       model=model, 
                       device=device, 
                       max_tokens=CLIP_TOKEN_LIMIT),
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

    output_json_dir = os.path.join(args.dataset, f"captions_blip2_rank{rank}.json")
    with open(output_json_dir, "w", encoding="utf-8") as jf:
        json.dump(all_descriptions, jf, indent=4, ensure_ascii=False)

    print(f"Process {rank} has saved BLIP2 descriptions to {output_json_dir}")