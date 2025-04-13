import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import torchvision
import argparse
import math

import subprocess
import webbrowser
import time
import sys
from threading import Thread

# Import models and hyperparameters
from models import DiffusionModel
from hparams import hparams, DiffusionHParams
from custom_dataset import CustomImageDataset

# Set up logging directory
log_dir = os.path.join("runs", f"diffusion_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
writer = SummaryWriter(log_dir)

# Extract value at a specific timestep
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    device = t.device
    a = a.to(device)
    
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# Forward diffusion process
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
        
    sqrt_alphas_cumprod_t = extract(hparams.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(hparams.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# Loss function - with self-conditioning support
def p_losses(denoise_model, x_start, t, noise=None, self_cond=None, use_self_conditioning=False):
    if noise is None:
        noise = torch.randn_like(x_start)
        
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    
    # Process self-conditioning (prediction from previous iteration)
    if use_self_conditioning and hasattr(denoise_model, 'use_self_conditioning'):
        predicted_noise = denoise_model(x_noisy, t, self_cond)
    else:
        predicted_noise = denoise_model(x_noisy, t)
    
    if predicted_noise.shape != noise.shape:
        predicted_noise = F.interpolate(
            predicted_noise, 
            size=noise.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
    
    loss = F.mse_loss(noise, predicted_noise)
    return loss, x_noisy, predicted_noise

# DDPM reverse diffusion process
@torch.no_grad()
def p_sample_ddpm(model, x, t, t_index, self_cond=None, use_self_conditioning=False):
    betas_t = extract(hparams.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(hparams.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(hparams.sqrt_recip_alphas, t, x.shape)
    
    # Model prediction with self-conditioning
    if use_self_conditioning and hasattr(model, 'use_self_conditioning'):
        predicted_noise = model(x, t, self_cond)
    else:
        predicted_noise = model(x, t)
    
    # Apply denoising formula
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(hparams.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# DDIM sampling (more efficient sampling)
@torch.no_grad()
def p_sample_ddim(model, x, t, t_prev, self_cond=None, use_self_conditioning=False, eta=0.0):
    """
    DDIM sampling step
    """
    device = x.device
    
    # Extract alpha_cumprod for current and previous timestep
    alpha_cumprod_t = extract(hparams.alphas_cumprod, t, x.shape)
    
    # Check if this is the final step (all batch elements have the same value so we check the first one)
    is_final_step = t_prev[0].item() < 0
    
    # Predict noise with self-conditioning
    if use_self_conditioning and hasattr(model, 'use_self_conditioning'):
        predicted_noise = model(x, t, self_cond)
    else:
        predicted_noise = model(x, t)
    
    # Extract x0: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1-alpha_cumprod) * ε
    pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
    
    # Clamp values for stability
    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    
    # Return predicted image directly at final step
    if is_final_step:
        return pred_x0
    
    # Apply DDIM formula for non-final steps
    alpha_cumprod_t_prev = extract(hparams.alphas_cumprod, t_prev, x.shape)
    
    # Calculate x_t direction
    dir_xt = torch.sqrt(1. - alpha_cumprod_t_prev - eta * eta * (1. - alpha_cumprod_t_prev) * (1. - alpha_cumprod_t) / (1. - alpha_cumprod_t_prev)) * predicted_noise
    
    # Add noise if eta > 0 (stochastic sampling)
    if eta > 0:
        noise = eta * torch.randn_like(x)
    else:
        noise = 0
    
    # Calculate x_{t-1}
    x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + noise
    
    return x_prev

# Unified sampling function for both DDPM and DDIM
@torch.no_grad()
def sample_diffusion(model, shape, sampling_method='ddpm', ddim_steps=100, ddim_eta=0.2, 
                     use_self_conditioning=False, epoch=0, num_images_to_log=10):
    device = next(model.parameters()).device
    
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []
    self_cond = None  # Initialize self-conditioning input
    
    # Log initial noise
    if num_images_to_log > 0:
        grid = torchvision.utils.make_grid(img[:num_images_to_log], normalize=True)
        writer.add_image(f'Diffusion Process/{sampling_method.upper()}/Initial Noise', grid, epoch)
    
    # DDIM sampling
    if sampling_method == 'ddim':
        timesteps = hparams.timesteps
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
            
            # Sample with DDIM and track self_cond
            img = p_sample_ddim(
                model, img, t, t_prev_tensor, 
                self_cond=self_cond, 
                use_self_conditioning=use_self_conditioning,
                eta=ddim_eta
            )
            
            # Update self_cond for next step if using self-conditioning
            if use_self_conditioning and hasattr(model, 'use_self_conditioning'):
                with torch.no_grad():
                    self_cond = img.detach().clone()
            
            if i % (max(ddim_steps // 10, 1)) == 0 or i == len(timesteps_array) - 1:
                print(f"DDIM Sampling step {i+1}/{len(timesteps_array)}, t={timestep}")
                if num_images_to_log > 0:
                    enhanced_img = enhance_contrast(img[:num_images_to_log])
                    grid = torchvision.utils.make_grid(enhanced_img, normalize=True)
                    writer.add_image(f'Diffusion Process/{sampling_method.upper()}/Step {i}', grid, epoch)
                imgs.append(img.cpu().numpy())
    else:
        # Original DDPM sampling
        log_timesteps = list(range(0, hparams.timesteps, hparams.timesteps // 10)) + [hparams.timesteps - 1]
        
        for i in reversed(range(0, hparams.timesteps)):
            # Sample with DDPM and track self_cond
            img = p_sample_ddpm(
                model, img, 
                torch.full((b,), i, device=device, dtype=torch.long), 
                i, 
                self_cond=self_cond,
                use_self_conditioning=use_self_conditioning
            )
            
            # Update self_cond for next step if using self-conditioning
            if use_self_conditioning and hasattr(model, 'use_self_conditioning'):
                with torch.no_grad():
                    self_cond = img.detach().clone()
            
            if i in log_timesteps and num_images_to_log > 0:
                enhanced_img = enhance_contrast(img[:num_images_to_log])
                grid = torchvision.utils.make_grid(enhanced_img, normalize=True)
                writer.add_image(f'Diffusion Process/{sampling_method.upper()}/Step {hparams.timesteps-i}', grid, epoch)
            
            if i % 100 == 0:
                print(f"DDPM Sampling timestep {i}")
                imgs.append(img.cpu().numpy())
    
    imgs.append(img.cpu().numpy())
    return imgs

# Image contrast enhancement function
def enhance_contrast(tensor_images, percentile_low=5, percentile_high=95):
    batch_size = tensor_images.shape[0]
    enhanced_images = []
    
    # Process each image in the batch
    for i in range(batch_size):
        img = tensor_images[i].clone()  # [C, H, W]
        
        # Process each channel
        for c in range(img.shape[0]):
            channel = img[c]
            
            # Calculate boundaries
            with torch.no_grad():
                sorted_vals = torch.sort(channel.flatten())[0]
                n_elements = sorted_vals.shape[0]
                low_idx = max(0, int(n_elements * percentile_low / 100))
                high_idx = min(n_elements - 1, int(n_elements * percentile_high / 100))
                
                low_val = sorted_vals[low_idx]
                high_val = sorted_vals[high_idx]
                
                # Expand value range (0~1)
                if high_val > low_val:
                    channel = torch.clamp((channel - low_val) / (high_val - low_val), 0, 1)
                    img[c] = channel
        
        enhanced_images.append(img)
    
    return torch.stack(enhanced_images)

# Visualize diffusion steps
def visualize_diffusion_steps(x_start, writer, num_steps=10):
    device = x_start.device
    
    step_indices = torch.linspace(0, hparams.timesteps-1, num_steps).long().to(device)
    
    noisy_images = []
    for i, t in enumerate(step_indices):
        t_batch = torch.full((x_start.shape[0],), t, device=device)
        noisy_image = q_sample(x_start, t_batch)
        noisy_images.append(noisy_image)
        
        # Apply contrast enhancement
        enhanced_image = enhance_contrast(noisy_image)
        grid = torchvision.utils.make_grid(enhanced_image, normalize=True)
        # writer.add_image(f'Forward Diffusion/Step {i}', grid, 0)
    
    all_images = torch.cat(noisy_images, dim=0)
    # Apply contrast enhancement to all images
    enhanced_all = enhance_contrast(all_images)
    grid = torchvision.utils.make_grid(enhanced_all, nrow=x_start.shape[0], normalize=True)
    writer.add_image('Forward Diffusion/All Steps', grid, 0)
    
    return noisy_images

# Create learning rate scheduler
def get_lr_scheduler(optimizer, num_training_steps):
    if hparams.lr_scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps - hparams.lr_warmup_steps
        )
    elif hparams.lr_scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_training_steps // 3,
            gamma=0.5
        )
    else:
        return None

# Create warmup learning rate scheduler
def get_warmup_scheduler(optimizer):
    def lr_lambda(current_step: int):
        if current_step < hparams.lr_warmup_steps:
            return float(current_step) / float(max(1, hparams.lr_warmup_steps))
        return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Create model from checkpoint function
def create_model_from_checkpoint(checkpoint_path, device, strict=False):
    """Create an identical model from checkpoint structure information"""
    print(f"Creating model from checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = None
    
    # Extract initialization args from checkpoint
    if '__init_args__' in checkpoint:
        model_args = checkpoint['__init_args__']
    elif 'model_init_args' in checkpoint:
        model_args = checkpoint['model_init_args']
    
    # If no args found, extract from hyperparameters
    if not model_args and 'hparams' in checkpoint:
        saved_hparams = checkpoint['hparams']
        model_args = {
            'image_channels': saved_hparams.get('image_channels', 1),
            'time_emb_dim': saved_hparams.get('time_emb_dim', 128),
            'base_channels': saved_hparams.get('base_channels', 64),
            'channel_mults': saved_hparams.get('channel_mults', (1, 2, 4, 8)),
            'num_res_blocks': saved_hparams.get('num_res_blocks', 2),
            'attention_levels': saved_hparams.get('attention_levels', (1, 2)),
            'dropout': saved_hparams.get('dropout', 0.1),
            'groups': saved_hparams.get('groups', 32),
            'use_self_conditioning': saved_hparams.get('use_self_conditioning', False),
            'image_size': saved_hparams.get('image_size', 28)
        }
    
    if model_args:
        # Convert tuple strings to actual tuples
        for key in ['channel_mults', 'attention_levels', 'attention_resolutions']:
            if key in model_args and isinstance(model_args[key], str):
                try:
                    # Convert string "(0, 1, 2)" to tuple (0, 1, 2)
                    if model_args[key].startswith('(') and model_args[key].endswith(')'):
                        # Remove parentheses, split by comma, and convert to integers
                        values = model_args[key].strip('()').split(',')
                        model_args[key] = tuple(int(val.strip()) for val in values if val.strip())
                except Exception as e:
                    print(f"Error converting {key}: {e}")
        
        print("Model parameters restored from checkpoint:")
        for key, value in model_args.items():
            print(f"  - {key}: {value}")
            
        # Create model
        model = DiffusionModel(**model_args)
        model.__init_args__ = model_args  # Store initialization args
        model = model.to(device)
        
        # Load weights partially (skip mismatched layers with strict=False)
        if strict:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Successfully loaded checkpoint weights completely")
            except Exception as e:
                print(f"Complete loading failed: {e}")
                return partial_load_weights(model, checkpoint['model_state_dict'], device), checkpoint
        else:
            return partial_load_weights(model, checkpoint['model_state_dict'], device), checkpoint
    
    print("Could not find model initialization info in checkpoint")
    return None, checkpoint

def partial_load_weights(model, state_dict, device):
    """Load only matching layers, skipping those with different structures"""
    print("Performing partial weight loading...")
    
    # Model's current state dictionary
    model_dict = model.state_dict()
    
    # Weights to load (only those with matching sizes)
    compatible_weights = {}
    incompatible_layers = []
    missing_layers = []
    unexpected_layers = []
    loaded_layers = []
    
    # Find weights that are in checkpoint and match the model
    for name, param in state_dict.items():
        if name in model_dict:
            if model_dict[name].shape == param.shape:
                compatible_weights[name] = param
                loaded_layers.append(name)
            else:
                incompatible_layers.append((name, param.shape, model_dict[name].shape))
        else:
            unexpected_layers.append(name)
    
    # Find layers in model that are not in checkpoint
    for name in model_dict:
        if name not in state_dict:
            missing_layers.append(name)
    
    # Load compatible weights
    model_dict.update(compatible_weights)
    model.load_state_dict(model_dict)
    
    # Print loading statistics
    print(f"Loaded layers: {len(loaded_layers)}/{len(model_dict)} ({len(loaded_layers) / len(model_dict) * 100:.1f}%)")
    print(f"Size mismatch layers: {len(incompatible_layers)}")
    print(f"Layers in model but not in checkpoint: {len(missing_layers)}")
    print(f"Layers in checkpoint but not in model: {len(unexpected_layers)}")
    
    # Print first few mismatched layers
    if incompatible_layers:
        print("\nExample mismatched layers:")
        for i, (name, cp_shape, model_shape) in enumerate(incompatible_layers[:5]):
            print(f"  {name}: checkpoint={cp_shape}, model={model_shape}")
        if len(incompatible_layers) > 5:
            print(f"  ... {len(incompatible_layers)} more")
    
    # Print first few missing layers
    if missing_layers:
        print("\nExample missing layers:")
        for name in missing_layers[:5]:
            print(f"  {name}")
        if len(missing_layers) > 5:
            print(f"  ... {len(missing_layers)} more")
    
    return model

# Checkpoint management function - simplified to remove EMA
def manage_checkpoints(log_dir, keep_last_n=3, model_prefix="model_epoch_"):
    # List all checkpoint files
    all_files = os.listdir(log_dir)
    
    # Get model checkpoints
    model_checkpoints = [f for f in all_files if f.startswith(model_prefix) and f.endswith(".pth")]
    
    # Sort checkpoints by epoch number (extract from filename)
    def get_epoch_num(filename, prefix):
        # Extract epoch number from filename (model_epoch_X.pth)
        return int(filename.replace(prefix, "").replace(".pth", ""))
    
    model_checkpoints.sort(key=lambda x: get_epoch_num(x, model_prefix))
    
    # Keep only the last N checkpoints, delete older ones
    checkpoints_to_delete = model_checkpoints[:-keep_last_n] if len(model_checkpoints) > keep_last_n else []
    
    # Delete older checkpoints
    for checkpoint in checkpoints_to_delete:
        try:
            os.remove(os.path.join(log_dir, checkpoint))
            print(f"Deleted older checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Error deleting checkpoint {checkpoint}: {e}")

# Run TensorBoard
def run_tensorboard(logdir="runs"):
    def start_tensorboard():
        try:
            if sys.platform.startswith('win'):
                cmd = f"tensorboard --logdir={logdir} --port=6006"
                subprocess.Popen(cmd, shell=True)
            else:
                cmd = ["tensorboard", "--logdir", logdir, "--port", "6006"]
                subprocess.Popen(cmd)
            
            time.sleep(3)
            webbrowser.open("http://localhost:6006")
            
        except Exception as e:
            print(f"TensorBoard execution error: {e}")
            print("Make sure TensorBoard is installed: pip install tensorboard")
    
    tb_thread = Thread(target=start_tensorboard)
    tb_thread.daemon = True
    tb_thread.start()
    
    print(f"TensorBoard is running at http://localhost:6006")
    return tb_thread

# Enhanced checkpoint saving function
def save_checkpoint(model, optimizer, epoch, avg_loss, log_dir, filename):
    """Enhanced checkpoint saving that includes model structure information"""
    # Extract model initialization args
    init_args = getattr(model, '__init_args__', None)
    
    # Use default values if no initialization args found
    if init_args is None:
        init_args = {
            'image_channels': hparams.image_channels, 
            'time_emb_dim': hparams.time_emb_dim,
            'base_channels': hparams.base_channels,
            'channel_mults': hparams.channel_mults,
            'num_res_blocks': hparams.num_res_blocks,
            'attention_levels': hparams.attention_levels,
            'dropout': hparams.dropout,
            'groups': hparams.groups,
            'use_self_conditioning': hparams.use_self_conditioning,
            'image_size': hparams.image_size
        }
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': avg_loss,
        'hparams': hparams.to_dict(),
        # Important: Store model initialization args - duplicate storage for compatibility
        '__init_args__': init_args,
        'model_init_args': init_args
    }
    
    torch.save(checkpoint, os.path.join(log_dir, filename))
    print(f"Checkpoint saved: {filename}")

# Training function - with gradient clipping, LR scheduling, and self-conditioning
def train(model, dataloader, optimizer, device, start_epoch=0):
    global_step = start_epoch * len(dataloader)
    
    # Get sample images for logging
    sample_images, _ = next(iter(dataloader))
    sample_images = sample_images[:8].to(device)
    
    # Set up model structure logging
    dummy_input = torch.randn(1, hparams.image_channels, hparams.image_size, hparams.image_size).to(device)
    dummy_time = torch.zeros(1, dtype=torch.long).to(device)
    
    # Check if model supports self-conditioning
    use_self_conditioning = hparams.use_self_conditioning and hasattr(model, 'use_self_conditioning')
    
    try:
        # Disable graph logging
        print("Skipping graph logging - not supported for dynamic models.")
    except Exception as e:
        print(f"Model graph logging error: {e}")
        print("Continuing training without graph logging.")
    
    # Visualize diffusion process
    visualize_diffusion_steps(sample_images, writer, num_steps=10)
    
    # Set up learning rate schedulers
    num_training_steps = len(dataloader) * hparams.epochs
    lr_scheduler = get_lr_scheduler(optimizer, num_training_steps)
    warmup_scheduler = get_warmup_scheduler(optimizer)
    
    for epoch in range(start_epoch, hparams.epochs):
        total_loss = 0
        num_batches = 0
        
        for step, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Random timesteps
            t = torch.randint(0, hparams.timesteps, (batch_size,), device=device).long()
            
            # Set up self-conditioning
            if use_self_conditioning:
                # First forward pass without self-conditioning
                with torch.no_grad():
                    x_noisy = q_sample(images, t)
                    self_cond = model(x_noisy, t)
                    # Ensure self_cond has the correct shape
                    if self_cond.shape != images.shape:
                        self_cond = F.interpolate(self_cond, size=images.shape[2:], 
                                                  mode='bilinear', align_corners=False)
                # Detach to prevent gradient propagation
                self_cond = self_cond.detach()
            else:
                self_cond = None
            
            # Calculate loss
            loss, noisy_images, predicted_noise = p_losses(
                model, images, t, 
                self_cond=self_cond, 
                use_self_conditioning=use_self_conditioning
            )
            total_loss += loss.item()
            num_batches += 1
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_norm)
            
            optimizer.step()
            
            # Update learning rate schedulers
            if global_step < hparams.lr_warmup_steps:
                warmup_scheduler.step()
            elif lr_scheduler is not None:
                lr_scheduler.step()
            
            # Log loss
            writer.add_scalar('Loss/train', loss.item(), global_step)
            
            if step % hparams.log_interval == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.6f}")
                
                # Log images
                if step % (hparams.log_interval * 5) == 0:
                    display_count = min(8, batch_size)
                    
                    # Original images - keep original without contrast adjustment
                    grid_original = torchvision.utils.make_grid(
                        images[:display_count], normalize=True)
                    writer.add_image('Original', grid_original, global_step)
                    
                    # Noisy images - apply contrast enhancement
                    enhanced_noisy = enhance_contrast(noisy_images[:display_count])
                    grid_noisy = torchvision.utils.make_grid(
                        enhanced_noisy, normalize=True)
                    writer.add_image('Noisy', grid_noisy, global_step)
                    
                    # Current denoising prediction
                    with torch.no_grad():
                        # Use model with self-conditioning if available
                        if use_self_conditioning:
                            current_prediction = model(noisy_images[:display_count], t[:display_count], self_cond[:display_count])
                        else:
                            current_prediction = model(noisy_images[:display_count], t[:display_count])
                        
                        # Resize prediction if needed
                        if current_prediction.shape != noisy_images.shape:
                            current_prediction = F.interpolate(
                                current_prediction, 
                                size=noisy_images.shape[2:],
                                mode='bilinear', 
                                align_corners=False
                            )
                        
                        sqrt_alphas_cumprod_t = extract(hparams.sqrt_alphas_cumprod, t[:display_count], 
                                                    noisy_images[:display_count].shape)
                        sqrt_one_minus_alphas_cumprod_t = extract(
                            hparams.sqrt_one_minus_alphas_cumprod, t[:display_count], noisy_images[:display_count].shape)
                        
                        # Calculate x0 prediction
                        predicted_x0 = (noisy_images[:display_count] - 
                                    sqrt_one_minus_alphas_cumprod_t * current_prediction) / sqrt_alphas_cumprod_t
                        
                        # Apply contrast enhancement
                        enhanced_predicted = enhance_contrast(predicted_x0)
                        grid_denoised = torchvision.utils.make_grid(
                            enhanced_predicted, normalize=True)
                        writer.add_image('Denoised', grid_denoised, global_step)
                
                # Log learning rate
                writer.add_scalar('Learning Rate', 
                                 optimizer.param_groups[0]['lr'], global_step)
                
                # Log gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('Gradients/norm', total_norm, global_step)
            
            global_step += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.6f}")
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        
        # Generate samples after epoch
        if epoch % hparams.sample_interval == 0 or epoch == hparams.epochs - 1:
            print(f"Generating samples for epoch {epoch}...")
            model.eval()
            
            # Use current model for sampling
            samples = sample_diffusion(
                model, 
                shape=(hparams.sample_batch_size, hparams.image_channels, hparams.image_size, hparams.image_size), 
                sampling_method=hparams.sampling_method,
                ddim_steps=hparams.ddim_steps,
                ddim_eta=hparams.ddim_sampling_eta,
                use_self_conditioning=use_self_conditioning,
                epoch=epoch, 
                num_images_to_log=hparams.sample_batch_size
            )

            final_samples = torch.from_numpy(samples[-1])
            enhanced_samples = enhance_contrast(final_samples)
            grid = torchvision.utils.make_grid(
                enhanced_samples, nrow=4, normalize=True)
            writer.add_image(f'Generated/Epoch{epoch}', grid, epoch)
            
            # Save checkpoint (including model initialization args)
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                log_dir, f"model_epoch_{epoch}.pth"
            )
            
            # Manage checkpoints (keep only the most recent 3)
            manage_checkpoints(log_dir, keep_last_n=3)
            
            model.train()
                
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Model")
    parser.add_argument('--data_dir', type=str, default=None, 
                        help='Directory containing custom dataset images (uses MNIST if not provided)')
    parser.add_argument('--image_size', type=int, default=hparams.image_size, 
                        help='Image size')
    parser.add_argument('--image_channels', type=int, default=hparams.image_channels, 
                        help='Number of image channels (1=grayscale, 3=RGB)')
    parser.add_argument('--batch_size', type=int, default=hparams.batch_size, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=hparams.epochs, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=hparams.learning_rate, 
                        help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=hparams.log_interval, 
                        help='Logging interval')
    parser.add_argument('--sample_interval', type=int, default=hparams.sample_interval, 
                        help='Sample generation interval (epochs)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Checkpoint file path to resume training from')
    parser.add_argument('--model_type', type=str, default='improved',
                        choices=['improved', 'advanced', 'complete'],
                        help='Model type to use: improved, advanced, or complete')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel count for the model')
    parser.add_argument('--use_self_conditioning', action='store_true',
                        help='Enable self-conditioning in the model')
    parser.add_argument('--use_mnist', action='store_true',
                        help='Force using MNIST dataset instead of custom dataset')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update hyperparameters - simplified version
    hparams.update(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        image_size=args.image_size,
        image_channels=args.image_channels,
        model_type=args.model_type,
        base_channels=args.base_channels,
        use_self_conditioning=args.use_self_conditioning
    )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize hyperparameters
    hparams.initialize_parameters(device=device)
    
    # Determine which dataset to use
    use_mnist = args.use_mnist or args.data_dir is None
    
    # Set up dataset and dataloader
    if use_mnist:
        print("Training using MNIST dataset")
        
        # Set default parameters for MNIST
        if args.use_mnist and args.image_size != 28:
            print(f"Warning: Resetting image size to 28 for MNIST (was: {args.image_size})")
            hparams.update(image_size=28)
            
        if args.use_mnist and args.image_channels != 1:
            print(f"Warning: Resetting image channels to 1 for MNIST (was: {args.image_channels})")
            hparams.update(image_channels=1)
            
        # Initialize MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        
    else:
        print(f"Using custom dataset: {args.data_dir}")
        
        try:
            dataset = CustomImageDataset(
                root_dir=args.data_dir,
                image_size=args.image_size,
                image_channels=args.image_channels,
                verbose=True  # For detailed logging
            )
            
            if len(dataset) == 0:
                raise ValueError(f"No valid images found in {args.data_dir}. Please check the path.")
            
            if len(dataset) < args.batch_size:
                print(f"Warning: Dataset size ({len(dataset)}) is smaller than batch size ({args.batch_size}).")
                print(f"Reducing batch size to {len(dataset)}")
                hparams.update(batch_size=len(dataset))
            
        except Exception as e:
            print(f"Error loading custom dataset: {str(e)}")
            print("Falling back to MNIST dataset")
            
            # Fallback to MNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            
            # Reset image size and channels to MNIST defaults
            hparams.update(image_size=28, image_channels=1)
    
    print(f"Dataset contains {len(dataset)} images")
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=hparams.batch_size, 
        shuffle=True,
        num_workers=0,  # Use num_workers=0 to prevent multiprocessing issues
        pin_memory=True if device == "cuda" else False
    )
    
    # Initialize model
    print(f"Initializing DiffusionModel...")
    model_init_args = {
        'image_channels': hparams.image_channels, 
        'time_emb_dim': hparams.time_emb_dim,
        'base_channels': hparams.base_channels,
        'channel_mults': hparams.channel_mults,
        'num_res_blocks': hparams.num_res_blocks,
        'attention_levels': hparams.attention_levels,
        'dropout': hparams.dropout,
        'groups': hparams.groups,
        'use_self_conditioning': hparams.use_self_conditioning,
        'image_size': hparams.image_size
    }
    
    model = DiffusionModel(**model_init_args)
    # Store initialization args
    model.__init_args__ = model_init_args
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay
    )
    
    # Try loading checkpoint if provided
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Attempting to load weights from checkpoint: {args.resume}")
        
        # Load checkpoint
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load weights partially
        model = partial_load_weights(model, checkpoint['model_state_dict'], device)
        
        # Try loading optimizer state
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Successfully loaded optimizer state")
        except Exception as e:
            print(f"Failed to load optimizer state (initializing new): {e}")
        
        # Set starting epoch
        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"Resuming training from checkpoint: Epoch {start_epoch}")
        
        # Restore hyperparameters if needed
        if 'hparams' in checkpoint:
            saved_hparams = checkpoint['hparams']
            for key, value in saved_hparams.items():
                if key not in ['epochs', 'learning_rate', 'batch_size'] and hasattr(hparams, key):
                    setattr(hparams, key, value)
            
            # Reinitialize hyperparameters
            hparams.initialize_parameters(device=device)
    
    # Check total epochs before training
    print(f"Total training epochs: {hparams.epochs}")
    if start_epoch > 0:
        print(f"Remaining epochs: {hparams.epochs - start_epoch}")
    
    # Run TensorBoard
    try:
        tb_thread = run_tensorboard(log_dir)
    except Exception as e:
        print(f"TensorBoard startup error: {e}")
        print("Continuing without TensorBoard.")
    
    # Log hyperparameters
    try:
        writer.add_hparams(
            hparams.to_dict(),
            {'hparam/placeholder': 0}
        )
    except Exception as e:
        print(f"Hyperparameter logging error: {e}")
    
    # Start training
    print(f"Starting training from epoch {start_epoch}...")
    model = train(model, dataloader, optimizer, device, start_epoch=start_epoch)
    
    # Save final model
    save_checkpoint(
        model, optimizer, hparams.epochs-1, 0.0,
        log_dir, 'diffusion_model_final.pth'
    )
    print(f"Model saved: {os.path.join(log_dir, 'diffusion_model_final.pth')}")
    
    # Close TensorBoard writer
    writer.close()
    print("Training complete. TensorBoard logs are available.")

if __name__ == "__main__":
    main()
