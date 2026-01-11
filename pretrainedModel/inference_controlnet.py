"""
ControlNet Inference Script

This script performs inference using a trained ControlNet model to generate images
from condition inputs (e.g., masks, edges, sketches).
"""

import argparse
import os
import cv2
import einops
import numpy as np
import torch
import random
from PIL import Image
from tqdm import tqdm

from share import *
import config
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def process_image(model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, 
                  num_samples, image_resolution, ddim_steps, guess_mode, 
                  strength, scale, seed, eta):
    """
    Process a single image through the ControlNet model.
    
    Args:
        model: ControlNet model
        ddim_sampler: DDIM sampler
        input_image: Input condition image (numpy array)
        prompt: Text prompt
        a_prompt: Additional prompt
        n_prompt: Negative prompt
        num_samples: Number of samples to generate
        image_resolution: Target resolution
        ddim_steps: Number of DDIM sampling steps
        guess_mode: Whether to use guess mode
        strength: Control strength
        scale: Guidance scale
        seed: Random seed (-1 for random)
        eta: DDIM eta parameter
        
    Returns:
        List of generated images (numpy arrays)
    """
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = HWC3(img)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control], 
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control], 
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # Set control scales
        if guess_mode:
            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)]
        else:
            model.control_scales = [strength] * 13

        # Sample
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples,
            shape, cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Decode
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='ControlNet Inference')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='./models/cldm_v15.yaml',
                        help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained ControlNet checkpoint')
    
    # Input/Output arguments
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input images directory (condition images)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output directory for generated images')
    
    # Generation arguments
    parser.add_argument('--prompt', type=str, default='',
                        help='Text prompt for generation')
    parser.add_argument('--a_prompt', type=str, default='',
                        help='Additional prompt (e.g., "best quality, detailed")')
    parser.add_argument('--n_prompt', type=str, default='',
                        help='Negative prompt (e.g., "lowres, bad quality")')
    
    # Sampling arguments
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate per input')
    parser.add_argument('--image_resolution', type=int, default=512,
                        help='Image resolution for generation')
    parser.add_argument('--ddim_steps', type=int, default=20,
                        help='Number of DDIM sampling steps')
    parser.add_argument('--guess_mode', action='store_true', default=False,
                        help='Enable guess mode (reduces control strength gradually)')
    parser.add_argument('--strength', type=float, default=1.0,
                        help='Control strength (0.0 to 2.0)')
    parser.add_argument('--scale', type=float, default=9.0,
                        help='Guidance scale (higher = more prompt adherence)')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed (-1 for random)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter (0.0 for deterministic)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_memory', action='store_true', default=False,
                        help='Enable memory optimization (for GPUs with < 12GB VRAM)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set memory optimization mode
    if args.save_memory:
        config.save_memory = True
        print('Memory optimization enabled (sliced attention + dynamic VRAM shifting)')
    else:
        config.save_memory = False
        print('Memory optimization disabled (faster inference)')
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    print(f'Output directory: {args.output_path}')
    
    # Load model
    print(f'Loading model from config: {args.config}')
    model = create_model(config_path=args.config).cpu()
    
    print(f'Loading checkpoint: {args.checkpoint}')
    model.load_state_dict(load_state_dict(args.checkpoint, location=args.device), strict=False)
    
    if args.device == 'cuda':
        model = model.cuda()
    
    # Create sampler
    ddim_sampler = DDIMSampler(model)
    print('Model loaded successfully')
    
    # Get input images
    if not os.path.exists(args.input_path):
        raise ValueError(f'Input path does not exist: {args.input_path}')
    
    image_files = [f for f in os.listdir(args.input_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if len(image_files) == 0:
        raise ValueError(f'No images found in {args.input_path}')
    
    print(f'Found {len(image_files)} images to process')
    
    # Process images
    print('Starting inference...')
    for image_name in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(args.input_path, image_name)
        
        # Load image
        image = Image.open(image_path)
        image = np.array(image)
        
        # Generate
        generated_images = process_image(
            model=model,
            ddim_sampler=ddim_sampler,
            input_image=image,
            prompt=args.prompt,
            a_prompt=args.a_prompt,
            n_prompt=args.n_prompt,
            num_samples=args.num_samples,
            image_resolution=args.image_resolution,
            ddim_steps=args.ddim_steps,
            guess_mode=args.guess_mode,
            strength=args.strength,
            scale=args.scale,
            seed=args.seed,
            eta=args.eta
        )
        
        # Save generated images
        for idx, gen_image in enumerate(generated_images):
            output_image = Image.fromarray(np.uint8(gen_image))
            
            # Create output filename
            name_without_ext = os.path.splitext(image_name)[0]
            if args.num_samples > 1:
                output_filename = f'{name_without_ext}_sample{idx}.png'
            else:
                output_filename = f'{name_without_ext}.png'
            
            output_filepath = os.path.join(args.output_path, output_filename)
            output_image.save(output_filepath)
    
    print(f'Inference completed! Generated images saved to: {args.output_path}')


if __name__ == '__main__':
    main()
