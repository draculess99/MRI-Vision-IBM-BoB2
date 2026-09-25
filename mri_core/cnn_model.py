"""Approximately 100k-parameter slice encoder and multilabel exam classifier."""

import torch
from torch import nn


class MRNetCNN(nn.Module):
    def __init__(self, num_planes=3, dropout=0.2, num_targets=3):
        super().__init__()
        if num_planes not in (1, 2, 3):
            raise ValueError("Expected one to three planes")
        if type(num_targets) is not int or num_targets < 1:
            raise ValueError("Expected a positive target count")
        self.num_planes = num_planes
        self.num_targets = num_targets
        blocks, previous = [], 1
        for channels in (16, 32, 64, 128):
            blocks.extend([nn.Conv2d(previous, channels, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, channels), nn.ReLU()])
            previous = channels
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128 * num_planes, num_targets))

    def forward(self, images, mask):
        if images.ndim != 6 or images.shape[1] != self.num_planes or images.shape[3] != 1:
            raise ValueError("Expected [batch, planes, slices, 1, height, width]")
        if mask.shape != images.shape[:3] or mask.dtype != torch.bool or not mask.any(dim=2).all():
            raise ValueError("Each exam plane requires at least one valid slice")
        # Padding never enters the encoder and cannot affect pooling or normalization.
        # Explicit spatial mean avoids nondeterministic CUDA adaptive-pooling backward kernels.
        embeddings = self.encoder(images[mask]).mean(dim=(-2, -1))
        expanded = embeddings.new_full((*mask.shape, 128), -torch.inf)
        expanded[mask] = embeddings
        pooled = expanded.max(dim=2).values.flatten(start_dim=1)
        return self.head(pooled)
