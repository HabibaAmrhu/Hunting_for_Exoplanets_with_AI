"""
LSTM-based model for exoplanet detection combining CNN feature extraction with BiLSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data."""
    
    def __init__(self, d_model: int, max_len: int = 2048):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
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
        return x + self.pe[:x.size(0), :]


class AttentionLayer(nn.Module):
    """Multi-head attention layer for temporal modeling."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize attention layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(AttentionLayer, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor with attention applied
        """
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection and residual connection
        output = self.w_o(context)
        return self.layer_norm(x + self.dropout(output))


class ExoplanetLSTM(nn.Module):
    """
    LSTM-based model combining CNN feature extraction with BiLSTM for temporal modeling.
    
    Architecture:
    1. CNN feature extraction layers
    2. Bidirectional LSTM for temporal modeling
    3. Attention mechanism for important time steps
    4. Classification head
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        sequence_length: int = 2048,
        cnn_channels: list = [32, 64, 128],
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        attention_heads: int = 8,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        use_positional_encoding: bool = True
    ):
        """
        Initialize LSTM-based exoplanet detection model.
        
        Args:
            input_channels: Number of input channels (1=raw, 2=raw+phase-folded)
            sequence_length: Length of input sequences
            cnn_channels: List of CNN channel sizes
            lstm_hidden_size: LSTM hidden state size
            lstm_num_layers: Number of LSTM layers
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanism
            use_positional_encoding: Whether to use positional encoding
        """
        super(ExoplanetLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding
        
        # CNN feature extraction layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(dropout_rate * 0.5)
            ))
            in_channels = out_channels
        
        # Calculate sequence length after CNN layers
        self.cnn_output_length = sequence_length
        for _ in cnn_channels:
            self.cnn_output_length = self.cnn_output_length // 2
        
        # Feature dimension after CNN
        self.feature_dim = cnn_channels[-1]
        
        # Positional encoding
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(self.feature_dim, self.cnn_output_length)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )
        
        # LSTM output dimension (bidirectional)
        self.lstm_output_dim = lstm_hidden_size * 2
        
        # Attention mechanism
        if self.use_attention:
            self.attention = AttentionLayer(
                d_model=self.lstm_output_dim,
                n_heads=attention_heads,
                dropout=dropout_rate
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_dim, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        features = x
        for cnn_layer in self.cnn_layers:
            features = cnn_layer(features)
        
        # Reshape for LSTM: (batch_size, seq_len, feature_dim)
        features = features.transpose(1, 2)  # (batch_size, seq_len, feature_dim)
        
        # Add positional encoding
        if self.use_positional_encoding:
            # Positional encoding expects (seq_len, batch_size, feature_dim)
            features_pe = features.transpose(0, 1)
            features_pe = self.pos_encoding(features_pe)
            features = features_pe.transpose(0, 1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Apply attention if enabled
        if self.use_attention:
            attended_features = self.attention(lstm_out)
        else:
            attended_features = lstm_out
        
        # Global average pooling over sequence dimension
        pooled_features = torch.mean(attended_features, dim=1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the model (before classification).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        features = x
        for cnn_layer in self.cnn_layers:
            features = cnn_layer(features)
        
        # Reshape for LSTM
        features = features.transpose(1, 2)
        
        # Add positional encoding
        if self.use_positional_encoding:
            features_pe = features.transpose(0, 1)
            features_pe = self.pos_encoding(features_pe)
            features = features_pe.transpose(0, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Apply attention if enabled
        if self.use_attention:
            attended_features = self.attention(lstm_out)
        else:
            attended_features = lstm_out
        
        # Global average pooling
        pooled_features = torch.mean(attended_features, dim=1)
        
        return pooled_features
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights if attention is enabled, None otherwise
        """
        if not self.use_attention:
            return None
        
        # This is a simplified version - in practice, you'd need to modify
        # the attention layer to return weights
        with torch.no_grad():
            batch_size = x.size(0)
            
            # CNN feature extraction
            features = x
            for cnn_layer in self.cnn_layers:
                features = cnn_layer(features)
            
            features = features.transpose(1, 2)
            
            if self.use_positional_encoding:
                features_pe = features.transpose(0, 1)
                features_pe = self.pos_encoding(features_pe)
                features = features_pe.transpose(0, 1)
            
            lstm_out, _ = self.lstm(features)
            
            # For visualization, return the mean attention across heads
            # This would need to be implemented in the attention layer
            return torch.ones(batch_size, lstm_out.size(1))  # Placeholder
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        return {
            'model_name': 'ExoplanetLSTM',
            'input_channels': self.input_channels,
            'sequence_length': self.sequence_length,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'use_attention': self.use_attention,
            'use_positional_encoding': self.use_positional_encoding,
            'total_parameters': self.count_parameters(),
            'cnn_output_length': self.cnn_output_length,
            'feature_dim': self.feature_dim
        }


class LightweightLSTM(nn.Module):
    """Lightweight LSTM model for faster training and inference."""
    
    def __init__(
        self,
        input_channels: int = 2,
        sequence_length: int = 2048,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        """
        Initialize lightweight LSTM model.
        
        Args:
            input_channels: Number of input channels
            sequence_length: Length of input sequences
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
        """
        super(LightweightLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Simple feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout1d(dropout_rate)
        )
        
        # Calculate feature dimension
        self.feature_length = sequence_length // 16  # Two maxpool layers with factor 4
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Feature extraction
        features = self.feature_extractor(x)
        features = features.transpose(1, 2)  # (batch, seq, features)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model info."""
        return {
            'model_name': 'LightweightLSTM',
            'input_channels': self.input_channels,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'total_parameters': self.count_parameters()
        }


def create_lstm_model(
    model_type: str = 'full',
    input_channels: int = 2,
    sequence_length: int = 2048,
    config: Optional[dict] = None
) -> nn.Module:
    """
    Factory function to create LSTM models.
    
    Args:
        model_type: Type of model ('full', 'lightweight')
        input_channels: Number of input channels
        sequence_length: Length of input sequences
        config: Optional model configuration
        
    Returns:
        LSTM model instance
    """
    if config is None:
        config = {}
    
    if model_type == 'full':
        default_config = {
            'cnn_channels': [32, 64, 128],
            'lstm_hidden_size': 128,
            'lstm_num_layers': 2,
            'attention_heads': 8,
            'dropout_rate': 0.3,
            'use_attention': True,
            'use_positional_encoding': True
        }
        default_config.update(config)
        
        return ExoplanetLSTM(
            input_channels=input_channels,
            sequence_length=sequence_length,
            **default_config
        )
    
    elif model_type == 'lightweight':
        default_config = {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout_rate': 0.2
        }
        default_config.update(config)
        
        return LightweightLSTM(
            input_channels=input_channels,
            sequence_length=sequence_length,
            **default_config
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")