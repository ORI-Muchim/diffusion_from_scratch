import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Time embedding using sinusoidal positions
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return embeddings

# Self-attention module
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, c, dim=1)
        
        q = q.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        
        scale = (c // self.num_heads) ** -0.5
        attention = torch.matmul(q, k) * scale
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.transpose(2, 3).reshape(b, c, h, w)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out + residual

# Group normalization with time embedding conditioning
class GroupNorm(nn.Module):
    def __init__(self, num_channels, time_embed_dim, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, num_channels)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, num_channels * 2)
        )
        
    def forward(self, x, time_embed):
        normalized = self.norm(x)
        params = self.mlp(time_embed)
        params = params.unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(params, 2, dim=1)
        
        return normalized * (1 + scale) + shift

# Residual block with time conditioning
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8, dropout=0.1, use_attention=False):
        super().__init__()
        
        self.norm1 = GroupNorm(in_channels, time_dim, groups)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = GroupNorm(out_channels, time_dim, groups)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention(out_channels, num_heads=4, dropout=dropout)
        
        self.gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
            nn.Sigmoid()
        )
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, time_emb):
        residual = self.skip(x)
        
        h = self.norm1(x, time_emb)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = self.norm2(h, time_emb)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        gate = self.gate(time_emb)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        h = h * gate
        
        if self.use_attention:
            h = self.attn(h)
        
        return h + residual

# Downsampling module
class Downsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        else:
            return F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

# Upsampling module
class Upsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.with_conv:
            return self.conv(x)
        return x

# Main diffusion model
class DiffusionModel(nn.Module):
    def __init__(
        self,
        image_channels=1,
        time_emb_dim=128,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=3,
        attention_levels=(1, 2),
        dropout=0.1,
        groups=8,
        use_self_conditioning=True,
        image_size=28
    ):
        super().__init__()
        self.use_self_conditioning = use_self_conditioning
        self.image_size = image_size
        self.image_channels = image_channels
        self.time_emb_dim = time_emb_dim * 4
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.GELU()
        )
        
        # Initial convolution
        in_ch = image_channels * 2 if use_self_conditioning else image_channels
        self.conv_in = nn.Conv2d(in_ch, base_channels, 3, padding=1)
        
        # Calculate channel dimensions for each level
        self.num_levels = len(channel_mults)
        self.channels_list = [base_channels]
        for i in range(self.num_levels):
            self.channels_list.append(base_channels * channel_mults[i])
        
        # Encoder modules (downsampling)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        in_out_channels = list(zip(self.channels_list[:-1], self.channels_list[1:]))
        
        # Track spatial dimensions
        self.feature_sizes = [image_size]
        current_size = image_size
        
        # Create encoder blocks
        for level, (in_ch, out_ch) in enumerate(in_out_channels):
            level_blocks = nn.ModuleList()
            
            for block_idx in range(num_res_blocks):
                if block_idx == 0:
                    level_blocks.append(
                        ResidualBlock(
                            in_ch, 
                            out_ch, 
                            self.time_emb_dim, 
                            groups, 
                            dropout,
                            use_attention=(level in attention_levels)
                        )
                    )
                else:
                    level_blocks.append(
                        ResidualBlock(
                            out_ch, 
                            out_ch, 
                            self.time_emb_dim, 
                            groups, 
                            dropout,
                            use_attention=(level in attention_levels)
                        )
                    )
            
            self.down_blocks.append(level_blocks)
            
            if level < len(in_out_channels) - 1:
                self.down_samples.append(Downsample(out_ch))
                current_size = current_size // 2
                self.feature_sizes.append(current_size)
        
        # Middle block (bottleneck)
        mid_channels = self.channels_list[-1]
        self.middle = nn.ModuleList([
            ResidualBlock(mid_channels, mid_channels, self.time_emb_dim, groups, dropout, use_attention=True),
            SelfAttention(mid_channels, num_heads=8, dropout=dropout),
            ResidualBlock(mid_channels, mid_channels, self.time_emb_dim, groups, dropout, use_attention=False)
        ])
        
        # Decoder modules (upsampling)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        reversed_channels = list(reversed(self.channels_list))
        
        # Create decoder blocks
        for level in range(self.num_levels):
            level_blocks = nn.ModuleList()
            
            if level == 0:
                in_ch = reversed_channels[level]
            else:
                in_ch = reversed_channels[level-1] + reversed_channels[level]
            
            out_ch = reversed_channels[level]
            
            if level > 0:
                self.up_samples.append(Upsample(reversed_channels[level-1]))
                current_size = current_size * 2
            
            for i in range(num_res_blocks + 1):
                if i == 0:
                    level_blocks.append(
                        ResidualBlock(
                            in_ch, 
                            out_ch, 
                            self.time_emb_dim, 
                            groups, 
                            dropout,
                            use_attention=(level in attention_levels)
                        )
                    )
                else:
                    level_blocks.append(
                        ResidualBlock(
                            out_ch, 
                            out_ch, 
                            self.time_emb_dim, 
                            groups, 
                            dropout,
                            use_attention=(level in attention_levels and i == 1)
                        )
                    )
            
            self.up_blocks.append(level_blocks)
        
        # Output blocks
        self.norm_out = GroupNorm(self.channels_list[0], self.time_emb_dim, groups)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.channels_list[0], self.channels_list[0] // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.channels_list[0] // 2, self.image_channels, 3, padding=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Weight initialization"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, t, self_cond=None):
        """Forward pass through the diffusion model"""
        # Handle self-conditioning
        if self.use_self_conditioning:
            if self_cond is None:
                self_cond = torch.zeros_like(x)
            x = torch.cat([x, self_cond], dim=1)
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Store skip connections
        skips = []
        
        # Encoder path (downsampling)
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            
            skips.append(h)
            
            if level < len(self.down_blocks) - 1:
                h = self.down_samples[level](h)
        
        # Middle processing (bottleneck)
        for block in self.middle:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:  # Attention block
                h = block(h)
        
        # Decoder path (upsampling)
        for level, blocks in enumerate(self.up_blocks):
            if level > 0:
                h = self.up_samples[level-1](h)
            
            if level > 0:
                skip = skips[-(level+1)]
                
                # Align spatial dimensions if needed
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
                
                h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                h = block(h, t_emb)
        
        # Final output processing
        h = self.norm_out(h, t_emb)
        h = self.act_out(h)
        
        # Resize to match input if needed
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return self.conv_out(h)
