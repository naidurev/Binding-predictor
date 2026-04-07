#!/usr/bin/env python3

import torch
import torch.nn as nn

class BindingSitePredictor(nn.Module):
    def __init__(self, input_dim=1291, hidden_dim=1280, num_heads=8, dropout=0.3, num_layers=3):
        super().__init__()

        # Normalize raw input features
        self.input_norm = nn.LayerNorm(input_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack num_layers transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        out = self.ffn(x)
        return out.squeeze(-1)


if __name__ == "__main__":
    model = BindingSitePredictor()
    x = torch.randn(2, 100, 1291)
    mask = torch.zeros(2, 100, dtype=torch.bool)

    out = model(x, mask)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
