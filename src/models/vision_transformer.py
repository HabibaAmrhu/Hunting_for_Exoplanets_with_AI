"""
Vision Transformer (ViT) adapted for 1D time series analysis.
Implements patch-based attention mechanism for light curve processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


class PatchEmbedding1D(nn.Module):
    """
    1D patch embedding for time series data.
    Converts 1D sequences into patch embeddings similar to image patches.
    """
    
    def __init__(
        self,
        sequence_length: int = 2048,
        patch_size: int = 16,
        in_channels: int = 2,
        embed_dim: int = 768
    ):
        """
        Initialize patch embedding.
        
        Args:
            sequence_length: Length of input sequence
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = sequence_length // patch_size
        
        # Patch embedding layer
        self.projection = nn.Conv1d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, in_channels, sequence_length)
            
        Returns:
            Patch embeddings (batch_size, num_patches, embed_dim)
        """
        # Apply patch embedding
        x = self.projection(x)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (batch_size, num_patches, embed_dim)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Final projection
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class ExoplanetViT(nn.Module):
    """
    Vision Transformer adapted for exoplanet detection in light curves.
    
    Uses patch-based attention mechanism to process 1D time series data.
    """
    
    def __init__(
        self,
        sequence_length: int = 2048,
        patch_size: int = 16,
        in_channels: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 1
    ):
        """
        Initialize Vision Transformer.
        
        Args:
            sequence_length: Length of input sequence
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Patch embedding
        self.patch_embed = PatchEmbedding1D(
            sequence_length, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, in_channels, sequence_length)
            
        Returns:
            Output tensor (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Classification (use class token)
        x = self.head(x[:, 0])  # (batch_size, num_classes)
        
        # Apply sigmoid for binary classification
        x = torch.sigmoid(x)
        
        return x
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps from all transformer blocks.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention maps from each block
        """
        batch_size = x.shape[0]
        attention_maps = []
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token and positional embedding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks and collect attention maps
        for block in self.blocks:
            # Get attention weights (simplified - would need modification to actual attention module)
            x = block(x)
            # In practice, you'd modify the attention module to return attention weights
            attention_maps.append(None)  # Placeholder
        
        return attention_maps
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        return {
            'model_name': 'ExoplanetViT',
            'sequence_length': self.sequence_length,
            'patch_size': self.patch_size,
            'in_channels': self.in_channels,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'total_parameters': self.count_parameters(),
            'num_patches': self.patch_embed.num_patches
        }


# Factory function
def create_exoplanet_vit(
    sequence_length: int = 2048,
    model_size: str = 'base',
    **kwargs
) -> ExoplanetViT:
    """
    Create ExoplanetViT model with predefined configurations.
    
    Args:
        sequence_length: Length of input sequence
        model_size: Model size ('tiny', 'small', 'base', 'large')
        **kwargs: Additional model parameters
        
    Returns:
        Configured ExoplanetViT model
    """
    configs = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'patch_size': 16
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'patch_size': 16
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'patch_size': 16
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'patch_size': 16
        }
    }
    
    config = configs.get(model_size, configs['base'])
    config.update(kwargs)
    
    return ExoplanetViT(
        sequence_length=sequence_length,
        **config
    )