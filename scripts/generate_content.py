#!/usr/bin/env python3
"""
generate_content.py - Script to generate content images with a consistent
character appearance across different contexts and scenarios.
"""

import os
import torch
import json
import argparse
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from datetime import datetime

class ContentGenerator:
    def __init__(self, args):
        self.args = args
        self.setup_directories()
        self.load_config()
        self.setup_model()
        
    def setup_directories(self):
        directories = [
            "assets/content_images/photoshoots",
            "assets/content_images/lifestyle",
            "assets/content_images/promotional",
            "output/instagram",
            "output/tiktok",
            "output/twitter",
            "config"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def load_config(self):
        try:
            with open(self.args.content_config, 'r') as f:
                self.content_config = json.load(f)
        except FileNotFoundError:
            self.content_config = {
                "scenarios": [
                    "influencer at coffee shop working on laptop",
                    "influencer at home office setup with modern tech gadgets",
                    "influencer at outdoor photoshoot in urban setting",
                    "influencer shopping at high-end fashion store",
                    "influencer at fitness studio in workout clothes"
                ],
                "styles": [
                    "professional photography, golden hour lighting",
                    "candid style, natural lighting",
                    "magazine editorial style, studio lighting",
                    "instagram aesthetic, vibrant colors",
                    "cinematic look, shallow depth of field"
                ],
                "aspect_ratios": {
                    "instagram": [1, 1],   # Square
                    "instagram_story": [9, 16],  # Vertical
                    "tiktok": [9, 16],     # Vertical
                    "twitter": [16, 9]     # Horizontal
                }
            }
            
            with open(self.args.content_config, 'w') as f:
                json.dump(self.content_config, f, indent=4)
    
    def setup_model(self):
        print("Loading model...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.args.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pipe.enable_attention_slicing()
    
    def load_character_prompt(self):
        try:
            with open(self.args.character_prompts, 'r') as f:
                character_data = json.load(f)
                self.character_prompt = character_data["base_character"]
                self.negative_prompt = character_data["negative_prompt"]
        except FileNotFoundError:
            print(f"Warning: Character prompts file not found: {self.args.character_prompts}")
            self.character_prompt = "petite female influencer with mixed Caucasian and Asian features, dark hair with blonde highlights"
            self.negative_prompt = "deformed, disfigured, poor quality, low quality, amateur"
    
    def load_reference_image(self):
        if self.args.reference_image:
            try:
                self.reference_image = Image.open(self.args.reference_image)
                print(f"Loaded reference image: {self.args.reference_image}")
                if self.reference_image.width > 1024 or self.reference_image.height > 1024:
                    self.reference_image.thumbnail((1024, 1024))
            except FileNotFoundError:
                print(f"Warning: Reference image not found: {self.args.reference_image}")
                self.reference_image = None
        else:
            self.reference_image = None
    
    def generate_content(self):
        self.load_character_prompt()
        self.load_reference_image()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = f"assets/content_images/session_{timestamp}"
        os.makedirs(session_folder, exist_ok=True)
        
        generated_images = []
        
        scenarios = self.content_config["scenarios"][:self.args.num_scenarios]
        styles = self.content_config["styles"][:self.args.num_styles]
        
        total_images = len(scenarios) * len(styles)
        current_image = 0
        
        for scenario_idx, scenario in enumerate(scenarios):
            for style_idx, style in enumerate(styles):
                current_image += 1
                print(f"Generating image {current_image}/{total_images}: {scenario} in {style}")
                
                full_prompt = f"{self.character_prompt}, {scenario}, {style}"
                
                seed = self.args.seed + (scenario_idx * 100) + style_idx
                generator = torch.Generator("cuda").manual_seed(seed)
                
                image = self.pipe(
                    prompt=full_prompt,
                    negative_prompt=self.negative_prompt,
                    num_inference_steps=self.args.steps,
                    guidance_scale=self.args.guidance_scale,
                    generator=generator
                ).images[0]
                
                scenario_name = scenario.replace(" ", "_")[:30]
                style_name = style.split(",")[0].replace(" ", "_")[:20]
                
                filename = f"{session_folder}/content_{scenario_name}_{style_name}_seed{seed}.png"
                image.save(filename)
                
                for platform, ratio in self.content_config["aspect_ratios"].items():
                    if platform.startswith(self.args.platform) or self.args.platform == "all":
                        platform_image = self.format_for_platform(image, ratio)
                        platform_filename = f"output/{platform.split('_')[0]}/{scenario_name}_{style_name}_seed{seed}.png"
                        platform_image.save(platform_filename)
                
                generated_images.append({
                    "filename": filename,
                    "prompt": full_prompt,
                    "seed": seed,
                    "scenario": scenario,
                    "style": style
                })
        
        report_file = f"{session_folder}/session_report.json"
        report = {
            "timestamp": timestamp,
            "model_id": self.args.model_id,
            "base_seed": self.args.seed,
            "steps": self.args.steps,
            "guidance_scale": self.args.guidance_scale,
            "character_prompts": self.args.character_prompts,
            "reference_image": self.args.reference_image,
            "generated_images": generated_images
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Generation complete! Generated {len(generated_images)} images.")
        print(f"Session report saved to: {report_file}")
    
    def format_for_platform(self, image, aspect_ratio):
        width, height = image.size
        target_ratio = aspect_ratio[0] / aspect_ratio[1]
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.1:
            if target_ratio > 1:  # Landscape
                new_size = (1200, int(1200 / target_ratio))
            else:  # Portrait or square
                new_size = (int(1200 * target_ratio), 1200)
            return image.resize(new_size, Image.LANCZOS)
        
        if current_ratio > target_ratio:
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            image = image.crop((left, 0, left + new_width, height))
        else:
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            image = image.crop((0, top, width, top + new_height))
        
        if target_ratio > 1:  # Landscape
            new_size = (1200, int(1200 / target_ratio))
        else:  # Portrait or square
            new_size = (int(1200 * target_ratio), 1200)
        
        return image.resize(new_size, Image.LANCZOS)

def main():
    parser = argparse.ArgumentParser(description="Generate AI influencer content images")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", 
                        help="Hugging Face model ID to use")
    parser.add_argument("--seed", type=int, default=1000, 
                        help="Base seed for generation")
    parser.add_argument("--steps", type=int, default=40, 
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, 
                        help="Guidance scale for prompt adherence")
    parser.add_argument("--character_prompts", type=str, default="prompts/character_prompts.json",
                        help="Path to character prompts file")
    parser.add_argument("--content_config", type=str, default="config/content_config.json",
                        help="Path to content generation config")
    parser.add_argument("--reference_image", type=str, default="",
                        help="Optional path to reference image for consistency")
    parser.add_argument("--num_scenarios", type=int, default=3,
                        help="Number of scenarios to generate")
    parser.add_argument("--num_styles", type=int, default=2,
                        help="Number of styles to apply")
    parser.add_argument("--platform", type=str, default="all",
                        choices=["all", "instagram", "tiktok", "twitter"],
                        help="Platform to format images for")
    
    args = parser.parse_args()
    
    generator = ContentGenerator(args)
    generator.generate_content()

if __name__ == "__main__":
    main()