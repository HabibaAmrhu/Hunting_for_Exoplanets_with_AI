"""
Transformer-based model for exoplanet detection with 1D positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding1D(nn.Module):
    """1D Positional encoding for time series data."""
    
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        """
        Initialize 1D positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding1D, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(context)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 8, 
        d_ff: int = 2048, 
        dropout: float = 0.1
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class ExoplanetTransformer(nn.Module):
    """
    Transformer-based model for exoplanet detection.
    
    Architecture:
    1. Input embedding/projection
    2. Positional encoding
    3. Multiple transformer blocks
    4. Global pooling
    5. Classification head
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        sequence_length: int = 2048,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout_rate: float = 0.1,
        max_seq_len: int = 2048,
        pooling_method: str = 'mean'
    ):
        """
        Initialize Transformer model.
        
        Args:
            input_channels: Number of input channels
            sequence_length: Length of input sequences
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout_rate: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            pooling_method: Pooling method ('mean', 'max', 'cls')
        """
        super(ExoplanetTransformer, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pooling_method = pooling_method
        
        # Input projection
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding1D(d_model, max_seq_len, dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Classification token (if using cls pooling)
        if pooling_method == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def create_padding_mask(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create padding mask for variable-length sequences.
        
        Args:
            x: Input tensor
            seq_lengths: Actual sequence lengths
            
        Returns:
            Padding mask
        """
        if seq_lengths is None:
            # No padding needed for fixed-length sequences
            return None
        
        batch_size, seq_len = x.size(0), x.size(1)
        mask = torch.arange(seq_len, device=x.device).expand(
            batch_size, seq_len
        ) < seq_lengths.unsqueeze(1)
        
        return mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
    
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            seq_lengths: Optional tensor of actual sequence lengths
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        batch_size, channels, seq_len = x.size()
        
        # Transpose to (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Input projection to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add CLS token if using cls pooling
        if self.pooling_method == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            seq_len += 1
        
        # Transpose for positional encoding (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transpose back to (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # Create attention mask
        mask = self.create_padding_mask(x, seq_lengths)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Pooling
        if self.pooling_method == 'cls':
            # Use CLS token
            pooled = x[:, 0, :]  # (batch_size, d_model)
        elif self.pooling_method == 'mean':
            # Global average pooling
            if seq_lengths is not None:
                # Masked average pooling
                mask_expanded = mask.squeeze(1).squeeze(1).unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = torch.mean(x, dim=1)
        elif self.pooling_method == 'max':
            # Global max pooling
            pooled, _ = torch.max(x, dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def get_features(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract features from the model (before classification).
        
        Args:
            x: Input tensor
            seq_lengths: Optional sequence lengths
            
        Returns:
            Feature tensor
        """
        batch_size, channels, seq_len = x.size()
        
        # Transpose and project
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        
        # Add CLS token if needed
        if self.pooling_method == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # Transformer blocks
        mask = self.create_padding_mask(x, seq_lengths)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Pooling
        if self.pooling_method == 'cls':
            pooled = x[:, 0, :]
        elif self.pooling_method == 'mean':
            if seq_lengths is not None:
                mask_expanded = mask.squeeze(1).squeeze(1).unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = torch.mean(x, dim=1)
        elif self.pooling_method == 'max':
            pooled, _ = torch.max(x, dim=1)
        
        return pooled
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention weights from a specific layer.
        
        Args:
            x: Input tensor
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Attention weights
        """
        # This is a simplified implementation
        # In practice, you'd need to modify the transformer blocks to return attention weights
        with torch.no_grad():
            batch_size, channels, seq_len = x.size()
            
            # Process through the network up to the specified layer
            x = x.transpose(1, 2)
            x = self.input_projection(x)
            
            if self.pooling_method == 'cls':
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                seq_len += 1
            
            x = x.transpose(0, 1)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)
            
            # Get attention weights from specified layer
            target_layer = self.transformer_blocks[layer_idx]
            attn_output, attn_weights = target_layer.attention(x, x, x)
            
            return attn_weights
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        return {
            'model_name': 'ExoplanetTransformer',
            'input_channels': self.input_channels,
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'pooling_method': self.pooling_method,
            'total_parameters': self.count_parameters()
        }


class LightweightTransformer(nn.Module):
    """Lightweight Transformer model for faster training."""
    
    def __init__(
        self,
        input_channels: int = 2,
        sequence_length: int = 2048,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout_rate: float = 0.1
    ):
        """
        Initialize lightweight transformer.
        
        Args:
            input_channels: Number of input channels
            sequence_length: Length of input sequences
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout_rate: Dropout rate
        """
        super(LightweightTransformer, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Downsample input first
        self.downsampler = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, d_model, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate downsampled length
        self.downsampled_length = sequence_length // 8
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding1D(d_model, self.downsampled_length, dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Downsample
        x = self.downsampler(x)  # (batch, d_model, downsampled_length)
        
        # Transpose for transformer
        x = x.transpose(1, 2)  # (batch, downsampled_length, d_model)
        
        # Positional encoding
        x = x.transpose(0, 1)  # (downsampled_length, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, downsampled_length, d_model)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        output = self.classifier(x)
        
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model info."""
        return {
            'model_name': 'LightweightTransformer',
            'input_channels': self.input_channels,
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'downsampled_length': self.downsampled_length,
            'total_parameters': self.count_parameters()
        }


def create_transformer_model(
    model_type: str = 'full',
    input_channels: int = 2,
    sequence_length: int = 2048,
    config: Optional[dict] = None
) -> nn.Module:
    """
    Factory function to create Transformer models.
    
    Args:
        model_type: Type of model ('full', 'lightweight')
        input_channels: Number of input channels
        sequence_length: Length of input sequences
        config: Optional model configuration
        
    Returns:
        Transformer model instance
    """
    if config is None:
        config = {}
    
    if model_type == 'full':
        default_config = {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1024,
            'dropout_rate': 0.1,
            'pooling_method': 'mean'
        }
        default_config.update(config)
        
        return ExoplanetTransformer(
            input_channels=input_channels,
            sequence_length=sequence_length,
            **default_config
        )
    
    elif model_type == 'lightweight':
        default_config = {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout_rate': 0.1
        }
        default_config.update(config)
        
        return LightweightTransformer(
            input_channels=input_channels,
            sequence_length=sequence_length,
            **default_config
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")