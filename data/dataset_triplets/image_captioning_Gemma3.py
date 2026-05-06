import os
import argparse
import json
import h5py
import torch
from PIL import Image
from transformers import pipeline
from datasets import Dataset
from functools import partial
from huggingface_hub import login

login(token="TOKEN")  # Replace with your Hugging Face token

# Model and Pipeline Setup
cache_dir = "/proj/berzelius-2023-364/users/x_gerna/model_cache/gemma3_cache"
os.makedirs(cache_dir, exist_ok=True)

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",   # e.g. google/gemma-3-12b-it, google/gemma-3-27b-it, etc.
    device="cuda",                 # or "cpu" if GPU is unavailable
    torch_dtype=torch.bfloat16,
    model_kwargs={"cache_dir": cache_dir}
)

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
def generate_captions_batch(examples, max_tokens=77):
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

# Main captioning function
def generate_image_captions_batched(
    images_dir,
    output_json,
    max_tokens,
    batch_size
):
    """
    1. Build a list of all images (from .hdf5 files).
    2. Convert that into a HuggingFace Dataset.
    3. Batch-load images (via dataset.map).
    4. Batch-generate captions (via dataset.map).
    5. Collect results in a dict -> JSON.
    """
    # Gather references to all images
    print(f"Creating image mapping from {images_dir}...")
    image_mapping = load_image_mapping(images_dir)
    print(f"Found {len(image_mapping)} images to process.")
    
    if not image_mapping:
        print("No images found! Exiting.")
        return
    
    # Create the dataset from the image mapping
    ds = Dataset.from_list(image_mapping)

    # Load images in batches
    ds = ds.map(
        load_images_in_batch,
        batched=True,
        batch_size=batch_size
    )

    # Generate captions in batches
    ds = ds.map(
        partial(generate_captions_batch, max_tokens=max_tokens),
        batched=True,
        batch_size=batch_size
    )

    # Convert to the desired dictionary structure
    all_descriptions = {}
    for item in ds:
        inst_id = item["instance_id"]
        img_key = item["image_key"]
        caption = item["caption"]
        if inst_id not in all_descriptions:
            all_descriptions[inst_id] = {}
        all_descriptions[inst_id][img_key] = caption

    # Write the results to JSON
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(all_descriptions, jf, indent=4, ensure_ascii=False)

    print(f"\nAll descriptions have been saved to {output_json}")


if __name__ == "__main__":
    CLIP_TOKEN_LIMIT = 77
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to the dataset")
    #parser.add_argument("--cache", help="Path to the Gemma 3 model cache directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for image loading and caption generation")
    
    args = parser.parse_args()
    
    images_dir = os.path.join(args.dataset, "images")
    output_dir = os.path.join(args.dataset, "captions.json")
    
    generate_image_captions_batched(
        images_dir=images_dir,
        output_json=output_dir,
        max_tokens=CLIP_TOKEN_LIMIT,
        batch_size=args.batch_size
    )
