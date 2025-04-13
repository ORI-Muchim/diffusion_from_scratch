import torch
import torch.nn.functional as F
import math

class DiffusionHParams:
    def __init__(self):
        # Base hyperparameters
        self.timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.012
        self.beta_schedule = 'cosine'  # 'linear' or 'cosine'
        
        # Training hyperparameters
        self.batch_size = 64
        self.epochs = 1000
        self.learning_rate = 5e-5
        self.log_interval = 100
        self.weight_decay = 1e-5
        self.lr_scheduler = 'cosine'  # 'cosine', 'step', or None
        self.lr_warmup_steps = 500
        self.grad_clip_norm = 1.0
        
        # Sampling parameters
        self.sampling_method = 'ddim'  # 'ddpm' or 'ddim'
        self.ddim_sampling_eta = 0.2    # 0 for deterministic sampling
        self.ddim_steps = 50
        
        # Image parameters
        self.image_size = 28  # For MNIST
        self.image_channels = 1  # Grayscale
        
        # Model hyperparameters
        self.time_emb_dim = 32
        self.base_channels = 128
        self.groups = 4  # For group normalization
        
        # Advanced model hyperparameters
        self.model_type = 'complete'
        self.channel_mults = (1, 2, 4, 8)  # Channel multipliers per resolution
        self.attention_resolutions = (8, 16)  # Resolutions for attention
        self.attention_levels = (0, 1, 2)  # Levels to apply attention
        self.num_res_blocks = 3
        self.dropout = 0.2
        self.use_self_conditioning = True
        
        # EMA parameters
        self.use_ema = False  # Exponential moving average
        self.ema_decay = 0.9999
        
        # Sampling hyperparameters
        self.sample_batch_size = 32
        self.sample_interval = 5
        
        # Computed parameters (initialized later)
        self.betas = None
        self.alphas = None
        self.alphas_cumprod = None
        self.alphas_cumprod_prev = None
        self.sqrt_recip_alphas = None
        self.sqrt_alphas_cumprod = None
        self.sqrt_one_minus_alphas_cumprod = None
        self.posterior_variance = None
        
        # Initialize parameters
        self.initialize_parameters()
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule from 'Improved Denoising Diffusion Probabilistic Models'"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def initialize_parameters(self, device='cpu'):
        """Initialize noise schedule and related parameters"""
        # Set beta schedule
        if self.beta_schedule == 'linear':
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=device)
        elif self.beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(self.timesteps).to(device)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        # Compute derived parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def update(self, **kwargs):
        """Update hyperparameters with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown hyperparameter: {key}")
        
        # Recalculate parameters
        self.initialize_parameters()
    
    def to_dict(self):
        """Convert hyperparameters to dictionary"""
        result = {
            # Basic parameters
            'timesteps': self.timesteps,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'beta_schedule': self.beta_schedule,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'log_interval': self.log_interval,
            'weight_decay': self.weight_decay,
            'lr_scheduler': self.lr_scheduler,
            'lr_warmup_steps': self.lr_warmup_steps,
            'grad_clip_norm': self.grad_clip_norm,
            'sampling_method': self.sampling_method,
            'ddim_sampling_eta': self.ddim_sampling_eta,
            'ddim_steps': self.ddim_steps,
            'image_size': self.image_size,
            'image_channels': self.image_channels,
            'time_emb_dim': self.time_emb_dim,
            'base_channels': self.base_channels,
            'groups': self.groups,
            'sample_batch_size': self.sample_batch_size,
            'sample_interval': self.sample_interval,
            
            # Advanced model parameters
            'model_type': self.model_type,
            'channel_mults': str(self.channel_mults),
            'attention_resolutions': str(self.attention_resolutions),
            'attention_levels': str(self.attention_levels),
            'num_res_blocks': self.num_res_blocks,
            'dropout': self.dropout,
            'use_self_conditioning': self.use_self_conditioning,
            
            # EMA parameters
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay
        }
        return result

# Create default hyperparameters instance
hparams = DiffusionHParams()
