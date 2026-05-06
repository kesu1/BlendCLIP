import requests
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

cache_dir = "./models_cache/"

# CAPTIONING
 
blip_processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-6.7b",
    cache_dir=cache_dir,
)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  # For 4-bit mode, use load_in_4bit=True instead.
    # You can add additional parameters if needed.
)

blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b",
    #quantization_config=quantization_config,
    cache_dir=cache_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

blip_model.to(device)

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image = Image.open("/home/thesis/thesis_projects/repos/ULIP-Outdoors/data/dataset_triplets/results/imgs_presentation/cropped_img_013.png")

prompt = "a photo of"
inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(device, torch.bfloat16)

captions = []
for cap_idx in range(10): 
    generated_ids = blip_model.generate(
        **inputs,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
        max_new_tokens=100
    )
    captions.append(generated_ids)

# Decode the generated captions
#print("Generated Captions:")
for idx in range(len(captions)):
    caption_decoded = blip_processor.batch_decode(captions[idx], skip_special_tokens=True)[0] 
    #print(f"{caption_decoded}")
    captions[idx] = caption_decoded

# RANKING

from transformers import CLIPProcessor, CLIPModel

# Load the CLIP-ViT-Large model and processor
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14",
    )

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    cache_dir=cache_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto")

clip_model.to(device)

# Prepare inputs for CLIP:
# Pass the same image and all candidate captions to the processor.
clip_inputs = clip_processor(
    text=captions,
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

# Get similarity scores; CLIP returns logits_per_image that contain the similarity between the image and each caption.
with torch.no_grad():
    clip_outputs = clip_model(**clip_inputs)

# The logits (or cosine similarity scores) are in outputs.logits_per_image (shape: [1, num_captions])
scores = clip_outputs.logits_per_image.squeeze(0)  # remove batch dimension

# Combine captions and their scores and sort
ranked = sorted(
    zip(captions, scores.to(torch.float32).cpu().numpy()), 
    key=lambda x: x[1], 
    reverse=True
)

print("\nRanked Captions (best to worst):")
for rank, (cap, score) in enumerate(ranked, 1):
    print(f"{rank}: Score {score:.4f} - {cap}")
    
final_caption = ranked[0][0]
print(f"Final caption: {final_caption}")