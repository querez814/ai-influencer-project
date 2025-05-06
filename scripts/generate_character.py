"""
generate_character.py - Script to generate base images for an AI influencer character
with specific features (petite, Caucasian with Asian features, highlighted hair).
"""

import os
import torch
import json
import argparse
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from datetime import datetime

if not torch.cuda.is_available():
    print("Warning: CUDA not available. CPU gonna be slow")

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "assets/reference_images",
        "assets/character_images/raw",
        "assets/character_images/selected",
        "prompts",
        "models",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def load_prompts(prompt_file):
    try:
        with open(prompt_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        default_prompts = {
            "base_character": "high quality photograph of a petite female influencer, mixed Caucasian and Asian features, shoulder-length dark hair with blonde highlights, natural makeup, friendly smile, fashionable casual outfit, standing confidently, professional photography, natural lighting, ultra detailed, photorealistic, 8k, high definition",
            "negative_prompt": "deformed, disfigured, poor quality, low quality, amateur, distorted proportions, blurry, unrealistic features, excessive makeup, cartoon, anime style, extra limbs, bad anatomy",
            "variations": [
                "close-up portrait of a petite female influencer with mixed Caucasian and Asian features and dark hair with blonde highlights",
                "full body shot of a petite female influencer with mixed Caucasian and Asian features and dark hair with blonde highlights",
                "three-quarter view of a petite female influencer with mixed Caucasian and Asian features and dark hair with blonde highlights"
            ]
        }
        
        with open(prompt_file, 'w') as f:
            json.dump(default_prompts, f, indent=4)
        
        return default_prompts

def generate_character_images(prompts, args):
    """Generate character images using the provided prompts"""
    print("Loading model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe.enable_attention_slicing()
    
    print("Generating base character image...")
    base_generator = torch.Generator("cuda").manual_seed(args.seed)
    
    base_image = pipe(
        prompt=prompts["base_character"],
        negative_prompt=prompts["negative_prompt"],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=base_generator
    ).images[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"assets/character_images/raw/base_character_{timestamp}_seed{args.seed}.png"
    base_image.save(base_filename)
    print(f"Saved base image: {base_filename}")
    
    if args.variations > 0:
        print(f"Generating {args.variations} variations...")
        for i in range(args.variations):
            variation_seed = args.seed + i + 1
            variation_generator = torch.Generator("cuda").manual_seed(variation_seed)
            
            prompt_index = min(i, len(prompts["variations"]) - 1)
            variation_prompt = prompts["variations"][prompt_index]
            
            variation_image = pipe(
                prompt=variation_prompt,
                negative_prompt=prompts["negative_prompt"],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=variation_generator
            ).images[0]
            
            variation_filename = f"assets/character_images/raw/character_variation_{timestamp}_seed{variation_seed}.png"
            variation_image.save(variation_filename)
            print(f"Saved variation {i+1}: {variation_filename}")
    
    print("Character generation complete!")
    return base_filename

def save_generation_config(args, base_image_path):
    config = {
        "model_id": args.model_id,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "variations": args.variations,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_image_path": base_image_path
    }
    
    config_filename = f"config/character_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Saved generation config to: {config_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate AI influencer character images")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", 
                        help="Hugging Face model ID to use")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed for reproducibility")
    parser.add_argument("--steps", type=int, default=50, 
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, 
                        help="Guidance scale for prompt adherence")
    parser.add_argument("--variations", type=int, default=3, 
                        help="Number of variations to generate")
    parser.add_argument("--prompt_file", type=str, default="prompts/character_prompts.json",
                        help="Path to JSON file with prompts")
    
    args = parser.parse_args()
    
    setup_directories()
    
    prompts = load_prompts(args.prompt_file)
    
    base_image_path = generate_character_images(prompts, args)
    
    save_generation_config(args, base_image_path)

if __name__ == "__main__":
    main()