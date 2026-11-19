"""A small patch-based Transformer for nine-channel UCI HAR signals."""

import torch
from torch import nn


class LightweightTimeSeriesTransformer(nn.Module):
    """Classify 128-step, nine-channel windows into six activities.

    Shape: (batch, 9, 128) -> 16 patches of 8 steps -> (batch, 6) logits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_channels = 9
        self.sequence_length = 128
        self.patch_size = 8
        self.num_patches = self.sequence_length // self.patch_size
        self.embedding_dim = 64

        self.patch_embedding = nn.Linear(
            self.input_channels * self.patch_size, self.embedding_dim
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embedding_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(self.embedding_dim),
        )
        self.classifier = nn.Linear(self.embedding_dim, 6)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 3 or signals.shape[1:] != (
            self.input_channels, self.sequence_length
        ):
            raise ValueError(
                f"Expected signals of shape (batch, 9, 128), got {tuple(signals.shape)}"
            )

        # (B, 9, 128) -> (B, 9, 16, 8) -> (B, 16, 72)
        patches = signals.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(
            signals.shape[0], self.num_patches, self.input_channels * self.patch_size
        )
        tokens = self.patch_embedding(patches) + self.position_embedding
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


if __name__ == "__main__":
    model = LightweightTimeSeriesTransformer()
    example = torch.zeros(2, 9, 128)
    logits = model(example)
    print(f"input={tuple(example.shape)}, logits={tuple(logits.shape)}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters())}")
