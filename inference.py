import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torchvision
from PIL import Image
import argparse
import math

# Import the DiffusionModel
from models import DiffusionModel
from hparams import DiffusionHParams, hparams

# Extract values at specific timesteps
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    device = t.device
    a = a.to(device)
    
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# DDPM denoising step with self-conditioning support
@torch.no_grad()
def p_sample_ddpm(model, x, t, t_index, diffusion_params, self_cond=None):
    device = x.device
    
    # Extract parameters
    betas_t = extract(diffusion_params['betas'], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x.shape)
    sqrt_recip_alphas_t = extract(diffusion_params['sqrt_recip_alphas'], t, x.shape)
    
    # Model supports self-conditioning
    predicted_noise = model(x, t, self_cond)
    
    # Device consistency check
    assert predicted_noise.device == x.device, "Model output and input tensor have different devices"
    
    # Predict original image
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(diffusion_params['posterior_variance'], t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# DDIM denoising step with self-conditioning support
@torch.no_grad()
def p_sample_ddim(model, x, t, t_prev, diffusion_params, self_cond=None, eta=0.0):
    device = x.device
    
    # Extract alpha_cumprod for current and previous timestep
    alpha_cumprod_t = extract(diffusion_params['alphas_cumprod'], t, x.shape)
    alpha_cumprod_t_prev = extract(diffusion_params['alphas_cumprod'], t_prev, x.shape) if t_prev >= 0 else torch.ones_like(alpha_cumprod_t)
    
    # Model supports self-conditioning
    predicted_noise = model(x, t, self_cond)
    
    # Extract x0: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1-alpha_cumprod) * ε
    pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
    
    # Clamp values between -1 and 1
    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    
    # Calculate x_t direction
    dir_xt = torch.sqrt(1. - alpha_cumprod_t_prev - eta * eta * (1. - alpha_cumprod_t_prev) * (1. - alpha_cumprod_t) / (1. - alpha_cumprod_t_prev)) * predicted_noise
    
    # Add noise for stochastic sampling (eta=0 for deterministic)
    if eta > 0:
        noise = eta * torch.randn_like(x)
    else:
        noise = 0
    
    # Calculate x_{t-1}
    x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + noise
    
    return x_prev

# Full sampling process (supports both DDPM and DDIM)
@torch.no_grad()
def sample_diffusion(model, diffusion_params, shape=(16, 1, 28, 28), device='cuda', 
                     return_intermediates=False, sampling_method='ddim', 
                     ddim_steps=100, ddim_eta=0.0):
    model.eval()
    
    b = shape[0]
    img = torch.randn(shape, device=device)
    self_cond = None  # Initialize self-conditioning
    
    intermediates = []
    
    # Reverse diffusion process
    timesteps = diffusion_params['timesteps']
    
    if sampling_method == 'ddim':
        print(f"Using DDIM sampling with {ddim_steps} steps (eta={ddim_eta})")
        skip = timesteps // ddim_steps
        timesteps_array = np.array(list(range(0, timesteps, skip)))
        timesteps_array = np.flip(timesteps_array)
        
        for i, timestep in enumerate(timesteps_array):
            if i == len(timesteps_array) - 1:
                t_prev = -1  # Final step
            else:
                t_prev = timesteps_array[i + 1]
                
            t = torch.full((b,), timestep, device=device, dtype=torch.long)
            t_prev_tensor = torch.full((b,), t_prev, device=device, dtype=torch.long)
            
            img = p_sample_ddim(model, img, t, t_prev_tensor, diffusion_params, 
                               self_cond=self_cond, eta=ddim_eta)
            
            # Save current image for self-conditioning
            if model.use_self_conditioning:
                self_cond = img.detach().clone()
            
            if return_intermediates and (i % (max(len(timesteps_array) // 10, 1)) == 0 or i == len(timesteps_array) - 1):
                intermediates.append((timestep, img.cpu().clone()))
            
            if i % (max(len(timesteps_array) // 10, 1)) == 0:
                print(f"DDIM Sampling step {i+1}/{len(timesteps_array)}, t={timestep}")
    else:
        print(f"Using DDPM sampling with {timesteps} steps")
        for i in reversed(range(0, timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            img = p_sample_ddpm(model, img, t, i, diffusion_params, self_cond=self_cond)
            
            # Save current image for self-conditioning
            if model.use_self_conditioning:
                self_cond = img.detach().clone()
            
            if return_intermediates and i % 100 == 0:
                intermediates.append((i, img.cpu().clone()))
            
            if i % 100 == 0:
                print(f"DDPM Sampling timestep {i}/{timesteps}")
    
    if return_intermediates:
        return img.cpu(), intermediates
    else:
        return img.cpu()

# Save generated images
def save_images(images, output_dir, prefix="generated"):
    os.makedirs(output_dir, exist_ok=True)
    
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        img = images[i, 0].numpy()  # [1, H, W] -> [H, W]
        
        # Normalize [-1, 1] -> [0, 1] -> [0, 255]
        img = (img + 1) / 2.0
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        image_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        Image.fromarray(img, mode='L').save(image_path)
    
    print(f"{batch_size} images saved to {output_dir}.")

# Visualize generation process
def visualize_generation_process(intermediates, num_images=4, figsize=(15, 10), 
                                save_path=None):
    num_steps = len(intermediates)
    
    # Sort from high timestep (noise) to low timestep (image)
    intermediates = sorted(intermediates, key=lambda x: x[0], reverse=True)
    
    fig = plt.figure(figsize=figsize)
    
    plt.suptitle("Diffusion Generation Process: Noise → Image", fontsize=20)
    
    # Add space for direction arrow
    height_ratios = [1] * num_steps + [0.5]
    gs = plt.GridSpec(num_steps+1, num_images, height_ratios=height_ratios)
    
    # Plot images
    for step_idx, (timestep, images) in enumerate(intermediates):
        for img_idx in range(min(num_images, images.shape[0])):
            ax = plt.subplot(gs[step_idx, img_idx])
            
            img = images[img_idx, 0].numpy()
            ax.imshow(img, cmap='gray')
            
            if step_idx == 0:
                ax.set_title(f"Image {img_idx+1}", fontsize=12)
            
            if img_idx == 0:
                if step_idx == 0:  # First row (noise)
                    label = f"t={timestep}(Noise)"
                elif step_idx == num_steps-1:  # Last row (final image)
                    label = f"t={timestep}(Final)"
                else:
                    label = f"t={timestep}"
                ax.set_ylabel(label, fontsize=11)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            if step_idx == 0:  # First row (noise)
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            elif step_idx == num_steps-1:  # Last row (final image)
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2)
    
    # Add direction arrow
    arrow_ax = plt.subplot(gs[num_steps, :])
    arrow_ax.set_xlim(0, 1)
    arrow_ax.set_ylim(0, 1)
    arrow_ax.axis('off')
    arrow_ax.annotate('Denoising direction', xy=(0.5, 0.2), xytext=(0.5, 0.8), 
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='fancy', color='black', linewidth=2),
                    ha='center', va='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()

# Create image grid
def create_image_grid(images, nrow=4):
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True, 
                                      padding=2).cpu()
    
    grid = grid.permute(1, 2, 0).numpy()
    
    return grid

# Load hyperparameters from checkpoint
def load_hparams_from_checkpoint(checkpoint):
    if 'hparams' in checkpoint:
        loaded_hparams = DiffusionHParams()
        for key, value in checkpoint['hparams'].items():
            try:
                setattr(loaded_hparams, key, value)
            except Exception as e:
                print(f"Warning: Could not set parameter {key}: {e}")
        return loaded_hparams
    else:
        print("No hyperparameters found in checkpoint. Using defaults.")
        return hparams

# Main inference function
def inference(model_path, output_dir=None, batch_size=16, num_batches=1, 
             device=None, visualize=True, seed=None, sampling_method=None,
             ddim_steps=None, ddim_eta=None):
    # Set reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_dir = f"output/generated_images_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load hyperparameters
    loaded_hparams = load_hparams_from_checkpoint(checkpoint)
    loaded_hparams.initialize_parameters(device=device)
    
    # Override sampling parameters if needed
    if sampling_method is not None:
        loaded_hparams.sampling_method = sampling_method
    if ddim_steps is not None:
        loaded_hparams.ddim_steps = ddim_steps
    if ddim_eta is not None:
        loaded_hparams.ddim_sampling_eta = ddim_eta
    
    # Convert strings to tuples
    channel_mults = getattr(loaded_hparams, 'channel_mults', "(1, 2, 4, 8)")
    attention_levels = getattr(loaded_hparams, 'attention_levels', "(1, 2)")
    
    if isinstance(channel_mults, str):
        channel_mults = eval(channel_mults)
    if isinstance(attention_levels, str):
        attention_levels = eval(attention_levels)
    
    # Initialize DiffusionModel
    model = DiffusionModel(
        image_channels=loaded_hparams.image_channels,
        time_emb_dim=loaded_hparams.time_emb_dim,
        base_channels=getattr(loaded_hparams, 'base_channels', 64),
        channel_mults=channel_mults,
        num_res_blocks=getattr(loaded_hparams, 'num_res_blocks', 2),
        attention_levels=attention_levels,
        dropout=getattr(loaded_hparams, 'dropout', 0.1),
        groups=getattr(loaded_hparams, 'groups', 8),
        use_self_conditioning=getattr(loaded_hparams, 'use_self_conditioning', False),
        image_size=loaded_hparams.image_size
    ).to(device)
    
    print("DiffusionModel initialized")
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Model loaded from epoch {epoch}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Trying to load with strict=False...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Model loaded with some missing or unexpected keys")
    else:
        try:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model directly: {e}")
            print("Inference may not work correctly")
    
    model.eval()
    
    # Prepare diffusion parameters
    diffusion_params = {
        'timesteps': loaded_hparams.timesteps,
        'betas': loaded_hparams.betas,
        'alphas': loaded_hparams.alphas,
        'alphas_cumprod': loaded_hparams.alphas_cumprod,
        'alphas_cumprod_prev': loaded_hparams.alphas_cumprod_prev,
        'sqrt_recip_alphas': loaded_hparams.sqrt_recip_alphas,
        'sqrt_alphas_cumprod': loaded_hparams.sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': loaded_hparams.sqrt_one_minus_alphas_cumprod,
        'posterior_variance': loaded_hparams.posterior_variance
    }
    
    # Save intermediates only for first batch
    save_intermediates = visualize
    
    # Print sampling method info
    sampling_method = loaded_hparams.sampling_method
    print(f"Sampling method: {sampling_method}")
    if sampling_method == 'ddim':
        print(f"DDIM settings: steps={loaded_hparams.ddim_steps}, eta={loaded_hparams.ddim_sampling_eta}")
    
    # Print self-conditioning info
    print(f"Using self-conditioning: {model.use_self_conditioning}")
    
    # Generate images
    for batch_idx in range(num_batches):
        print(f"\nGenerating batch {batch_idx+1}/{num_batches}...")
        
        if batch_idx == 0 and save_intermediates:
            images, intermediates = sample_diffusion(
                model, diffusion_params, 
                shape=(batch_size, loaded_hparams.image_channels, loaded_hparams.image_size, loaded_hparams.image_size), 
                device=device,
                return_intermediates=True,
                sampling_method=sampling_method,
                ddim_steps=loaded_hparams.ddim_steps, 
                ddim_eta=loaded_hparams.ddim_sampling_eta
            )
            
            if visualize:
                vis_path = os.path.join(output_dir, "generation_process.png")
                visualize_generation_process(intermediates, save_path=vis_path)
        else:
            images = sample_diffusion(
                model, diffusion_params, 
                shape=(batch_size, loaded_hparams.image_channels, loaded_hparams.image_size, loaded_hparams.image_size), 
                device=device,
                return_intermediates=False,
                sampling_method=sampling_method,
                ddim_steps=loaded_hparams.ddim_steps, 
                ddim_eta=loaded_hparams.ddim_sampling_eta
            )
        
        # Save images
        batch_dir = os.path.join(output_dir, f"batch_{batch_idx+1}")
        save_images(images, batch_dir, prefix="generated")
        
        # Visualize grid
        if visualize:
            plt.figure(figsize=(10, 10))
            grid = create_image_grid(images, nrow=4)
            plt.imshow(grid, cmap='gray')
            plt.axis('off')
            plt.title(f"Generated Images (Batch {batch_idx+1})")
            
            grid_path = os.path.join(output_dir, f"grid_batch_{batch_idx+1}.png")
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"\nAll images generated in {output_dir}.")
    return output_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with DiffusionModel")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images per batch")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to generate")
    parser.add_argument("--no_visualization", action="store_true", help="Skip visualization")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--sampling_method", type=str, default=None, choices=['ddpm', 'ddim'],
                        help="Sampling method to use")
    parser.add_argument("--ddim_steps", type=int, default=None, 
                        help="Number of DDIM sampling steps (fewer = faster)")
    parser.add_argument("--ddim_eta", type=float, default=None, 
                        help="DDIM eta parameter (0: deterministic, 1: stochastic)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    inference(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=args.device,
        visualize=not args.no_visualization,
        seed=args.seed,
        sampling_method=args.sampling_method,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta
    )
