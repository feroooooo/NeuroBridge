import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import math


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)


class EEGNet(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()

        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (channels_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                nn.Dropout2d(0.5)
            )
        
        # Use a dummy tensor to pass through the backbone to calculate the flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels_num, eeg_sample_points)
            out = self.backbone(dummy)
            embedding_dim = out.shape[1] * out.shape[2] * out.shape[3]
        
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.backbone(x)
        x = x.view(x.size(0), -1) 
        x = self.project(x)
        return x

class EEGProject(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        self.input_dim = eeg_sample_points * channels_num

        self.model = nn.Sequential(nn.Linear(self.input_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.3),
            )),
            nn.LayerNorm(feature_dim))
        
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class TSConv(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        
        emb_size = 40
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))
        
        embedding_dim = (math.ceil((((eeg_sample_points - 25) + 1) - 51) / 5.) + 1) * 40
        self.proj_eeg = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.5),
            )),
            nn.LayerNorm(feature_dim),
        )
    
    def forward(self, x:Tensor):
        x = x.unsqueeze(dim=1)
        x = self.tsconv(x)
        x = self.projection(x)
        x = x.view(x.size(0), -1)
        x = self.proj_eeg(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class EEGTransformer(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        d_model = 128
        nhead = 8
        num_layers = 4
        dim_feedforward = 512
        dropout = 0.1
        
        # Project input (channels) -> embedding dimension
        self.input_proj = nn.Linear(channels_num, d_model)
        # Positional encoding across time dimension
        self.pos_encoder = PositionalEncoding(d_model, eeg_sample_points)
        # Transformer encoder (batch_first=True for [B, S, D])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Final projection to desired output dimension
        self.fc_out = nn.Linear(d_model, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG data tensor of shape [batch_size, channels_num, seq_len].
        Returns:
            Tensor of shape [batch_size, output_dim].
        """
        # Rearrange to [batch_size, seq_len, channels_num]
        x = x.permute(0, 2, 1)
        # Project to embedding dimension
        x = self.input_proj(x)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Transformer encoding
        x = self.transformer_encoder(x)
        # Pool across time (mean pooling)
        x = x.mean(dim=1)  # [batch_size, d_model]
        # Final feature projection
        x = self.fc_out(x)  # [batch_size, output_dim]
        return x

if __name__ == "__main__":
    # Example usage
    eeg_sample_points = 250
    channels_num = 17
    feature_dim = 1024
    model = EEGTransformer(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    
    # Create a dummy EEG input tensor with shape (batch_size, channels_num, eeg_sample_points)
    batch_size = 8
    dummy_eeg_input = torch.randn(batch_size, channels_num, eeg_sample_points)
    
    # Forward pass through the model
    output = model(dummy_eeg_input)
    print(output.shape)  # Expected output shape: (batch_size, feature_dim)