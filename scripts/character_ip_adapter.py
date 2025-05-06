#!/usr/bin/env python3
"""
character_ip_adapter.py - Script to use IP-Adapter for generating content
with a consistent character appearance.

This requires IP-Adapter to be installed. You can install it with:
pip install git+https://github.com/tencent-ailab/IP-Adapter.git

You'll also need to download the IP-Adapter weights.
"""

import os
import torch
import json
import argparse
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from datetime import datetime
from tqdm import tqdm

try:
    from ip_adapter import IPAdapterXL
except ImportError:
    print("Error: IP-Adapter package not found.")
    print("Please install it with: pip install git+https://github.com/tencent-ailab/IP-Adapter.git")
    exit(1)

class CharacterIPAdapter:
    def __init__(self, args):
        self.args = args
        self.setup_directories()
        self.load_config()
        
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            "assets/content_images/ip_adapter",
            "models/ip_adapter_weights",
            "output/instagram",
            "output/tiktok", 
            "output/twitter"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def load_config(self):
        """Load content generation configuration"""
        try:
            with open(self.args.content_config, 'r') as f:
                self.content_config = json.load(f)
        except FileNotFoundError:
            print(f"Error: Content config file not found: {self.args.content_config}")
            exit(1)
    
    def setup_model(self):
        """Setup the SDXL model with IP-Adapter"""
        print("Loading model...")
        
        # Check if IP-Adapter weights exist
        if not os.path.exists(self.args.adapter_weights):
            print(f"Error: IP-Adapter weights not found at {self.args.adapter_weights}")
            print("Please download the weights and provide the correct path.")
            exit(1)
        
        # Load the base model
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.args.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enable attention slicing and memory optimization
        pipe.enable_attention_slicing()
        
        # Setup IP-Adapter
        print("Setting up IP-Adapter...")
        self.ip_adapter = IPAdapterXL(pipe, self.args.adapter_weights, device="cuda")
        print("Model and IP-Adapter loaded successfully!")
    
    def load_reference_image(self):
        """Load and validate the reference character image"""
        if not os.path.exists(self.args.reference_image):
            print(f"Error: Reference image not found: {self.args.reference_image}")
            exit(1)
        
        try:
            self.reference_image = Image.open(self.args.reference_image)
            print(f"Loaded reference image: {self.args.reference_image}")
            
            # Resize if needed
            if max(self.reference_image.size) > 1024:
                self.reference_image.thumbnail((1024, 1024))
                print("Reference image resized to fit within 1024x1024")
        except Exception as e:
            print(f"Error loading reference image: {e}")
            exit(1)
    
    def load_prompts(self):
        """Load scenario prompts from content config"""
        self.scenarios = []
        
        # Mix scenarios with different styles from config
        scenarios = self.content_config["scenarios"][:self.args.num_scenarios]
        styles = self.content_config["styles"][:self.args.num_styles]
        
        for scenario in scenarios:
            for style in styles:
                self.scenarios.append({
                    "scenario": scenario,
                    "style": style,
                    "prompt": f"{scenario}, {style}"
                })
        
        print(f"Loaded {len(self.scenarios)} scenario-style combinations")
    
    def generate_content(self):
        """Generate content images with consistent character appearance"""
        self.setup_model()
        self.load_reference_image()
        self.load_prompts()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = f"assets/content_images/ip_adapter/session_{timestamp}"
        os.makedirs(session_folder, exist_ok=True)
        
        # Track all generated images
        generated_images = []
        
        print(f"Generating {len(self.scenarios)} images with consistent character appearance...")
        for i, scenario_data in enumerate(tqdm(self.scenarios)):
            prompt = scenario_data["prompt"]
            full_prompt = f"{self.args.character_prompt}, {prompt}"
            
            # Ensure deterministic generation with seed
            seed = self.args.seed + i
            generator = torch.Generator("cuda").manual_seed(seed)
            
            # Generate image with IP-Adapter
            image = self.ip_adapter.generate(
                pil_image=self.reference_image,
                prompt=full_prompt,
                negative_prompt=self.args.negative_prompt,
                scale=self.args.adapter_scale,  # IP-Adapter conditioning scale
                num_samples=1,
                seed=seed,
                guidance_scale=self.args.guidance_scale,
                num_inference_steps=self.args.steps,
                generator=generator
            )[0]
            
            # Format scenario info for filename
            scenario_name = scenario_data["scenario"].replace(" ", "_")[:30]
            style_name = scenario_data["style"].split(",")[0].replace(" ", "_")[:20]
            
            # Save the generated image
            filename = f"{session_folder}/character_{scenario_name}_{style_name}_seed{seed}.png"
            image.save(filename)
            print(f"Saved image {i+1}/{len(self.scenarios)}: {filename}")
            
            # Create formatted versions for each platform
            for platform, ratio in self.content_config["aspect_ratios"].items():
                if platform.startswith(self.args.platform) or self.args.platform == "all":
                    platform_image = self.format_for_platform(image, ratio)
                    platform_filename = f"output/{platform.split('_')[0]}/{scenario_name}_{style_name}_seed{seed}.png"
                    platform_image.save(platform_filename)
            
            # Add to tracking
            generated_images.append({
                "filename": filename,
                "prompt": full_prompt,
                "seed": seed,
                "scenario": scenario_data["scenario"],
                "style": scenario_data["style"],
                "adapter_scale": self.args.adapter_scale
            })
        
        # Generate session report
        report_file = f"{session_folder}/session_report.json"
        report = {
            "timestamp": timestamp,
            "model_id": self.args.model_id,
            "reference_image": self.args.reference_image,
            "character_prompt": self.args.character_prompt,
            "base_seed": self.args.seed,
            "steps": self.args.steps,
            "guidance_scale": self.args.guidance_scale,
            "adapter_scale": self.args.adapter_scale,
            "adapter_weights": self.args.adapter_weights,
            "generated_images": generated_images
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Generation complete! Generated {len(generated_images)} images.")
        print(f"Session report saved to: {report_file}")
    
    def format_for_platform(self, image, aspect_ratio):
        """Format image for specific platform aspect ratio"""
        width, height = image.size
        target_ratio = aspect_ratio[0] / aspect_ratio[1]
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.1:
            # Close enough to target ratio, just resize
            if target_ratio > 1:  # Landscape
                new_size = (1200, int(1200 / target_ratio))
            else:  # Portrait or square
                new_size = (int(1200 * target_ratio), 1200)
            return image.resize(new_size, Image.LANCZOS)
        
        # Need to crop to achieve target ratio
        if current_ratio > target_ratio:
            # Image is wider than needed
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            image = image.crop((left, 0, left + new_width, height))
        else:
            # Image is taller than needed
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            image = image.crop((0, top, width, top + new_height))
        
        # Resize to standard dimensions
        if target_ratio > 1:  # Landscape
            new_size = (1200, int(1200 / target_ratio))
        else:  # Portrait or square
            new_size = (int(1200 * target_ratio), 1200)
        
        return image.resize(new_size, Image.LANCZOS)

def main():
    parser = argparse.ArgumentParser(description="Generate content with consistent character using IP-Adapter")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", 
                        help="Hugging Face model ID to use")
    parser.add_argument("--reference_image", type=str, required=True,
                        help="Path to reference character image")
    parser.add_argument("--adapter_weights", type=str, 
                        default="models/ip_adapter_weights/ip_adapter_xl_sdxl.bin",
                        help="Path to IP-Adapter weights")
    parser.add_argument("--character_prompt", type=str, 
                        default="petite female influencer with mixed Caucasian and Asian features, dark hair with blonde highlights",
                        help="Base description of the character")
    parser.add_argument("--negative_prompt", type=str,
                        default="deformed, disfigured, poor quality, low quality, amateur, distorted proportions",
                        help="Negative prompt to avoid unwanted features")
    parser.add_argument("--content_config", type=str, default="config/content_config.json",
                        help="Path to content generation config")
    parser.add_argument("--seed", type=int, default=2000,
                        help="Base seed for generation")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="Guidance scale for prompt adherence")
    parser.add_argument("--adapter_scale", type=float, default=0.8,
                        help="Scale for IP-Adapter conditioning (0.0-1.0)")
    parser.add_argument("--num_scenarios", type=int, default=5,
                        help="Number of scenarios to generate")
    parser.add_argument("--num_styles", type=int, default=2,
                        help="Number of styles to apply")
    parser.add_argument("--platform", type=str, default="all",
                        choices=["all", "instagram", "tiktok", "twitter"],
                        help="Platform to format images for")
    
    args = parser.parse_args()
    
    adapter = CharacterIPAdapter(args)
    adapter.generate_content()

if __name__ == "__main__":
    main()