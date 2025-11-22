"""
Normalizing flow components for modeling joint distributions between fMRI features and image embeddings.

The implementation relies on affine coupling layers (RealNVP-style) with optional conditioning vectors
for cross-modal alignment. The conditioning setup allows us to train two symmetric flows:
- fMRI -> image embedding (condition on fMRI, predict image latent)
- image -> fMRI embedding (condition on image, predict fMRI latent)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


class MLP(nn.Sequential):
    """A lightweight MLP for scale/shift prediction inside coupling layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


class AffineCoupling(nn.Module):
    """
    RealNVP-style affine coupling layer with optional conditioning.

    A subset of the inputs is left unchanged (the "identity" channel) while the complement is
    transformed based on a learned scale and shift produced by an MLP that sees the identity
    split and an optional conditioning vector (e.g., from the opposite modality).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mask: Tensor,
        conditioning_dim: Optional[int] = None,
        clamp: float = 5.0,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        scale_shift_in = input_dim + (conditioning_dim or 0)
        self.nn = MLP(scale_shift_in, hidden_dim, input_dim * 2)
        self.clamp = clamp

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        masked_x = x * self.mask
        nn_in = masked_x
        if cond is not None:
            nn_in = torch.cat([nn_in, cond], dim=-1)
        shift, log_scale = self.nn(nn_in).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale / self.clamp) * self.clamp
        y = masked_x + (1 - self.mask) * (x * torch.exp(log_scale) + shift)
        log_det = ((1 - self.mask) * log_scale).sum(dim=-1)
        return y, log_det

    def inverse(self, y: Tensor, cond: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        masked_y = y * self.mask
        nn_in = masked_y
        if cond is not None:
            nn_in = torch.cat([nn_in, cond], dim=-1)
        shift, log_scale = self.nn(nn_in).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale / self.clamp) * self.clamp
        x = masked_y + (1 - self.mask) * ((y - shift) * torch.exp(-log_scale))
        log_det = -((1 - self.mask) * log_scale).sum(dim=-1)
        return x, log_det


def checkerboard_mask(dim: int, even: bool = True, device: Optional[torch.device] = None) -> Tensor:
    """Create a simple alternating binary mask for coupling layers."""

    mask = torch.zeros(dim, device=device)
    mask[::2] = 1 if even else 0
    mask[1::2] = 0 if even else 1
    return mask


@dataclass
class FlowStep:
    coupling: AffineCoupling
    actnorm: nn.Module
    permute: Optional[nn.Module] = None

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x, log_det = self.actnorm(x)
        if self.permute is not None:
            x = self.permute(x)
        x, c_log_det = self.coupling(x, cond)
        return x, log_det + c_log_det

    def inverse(self, y: Tensor, cond: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        y, c_log_det = self.coupling.inverse(y, cond)
        if self.permute is not None:
            y = self.permute.inverse(y)
        y, log_det = self.actnorm.inverse(y)
        return y, log_det + c_log_det


class ActNorm(nn.Module):
    """Data-dependent activation normalization as used in Glow."""

    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.initialized = False
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features))

    def initialize_parameters(self, x: Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + self.eps
            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(torch.log(1 / std))
        self.initialized = True

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.initialized:
            self.initialize_parameters(x)
        y = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum() * torch.ones(x.size(0), device=x.device)
        return y, log_det

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        x = y * torch.exp(-self.log_scale) - self.bias
        log_det = -self.log_scale.sum() * torch.ones(y.size(0), device=y.device)
        return x, log_det


class RandomPermutation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        perm = torch.randperm(dim)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x: Tensor) -> Tensor:
        return x[:, self.perm]

    def inverse(self, y: Tensor) -> Tensor:
        return y[:, self.inv_perm]


class ConditionalRealNVP(nn.Module):
    """
    Stacked RealNVP blocks for conditional density estimation.

    The design mirrors Glow/RealNVP, using ActNorm + (optional) permutation + affine coupling.
    Conditioning vectors enable cross-modal modeling between fMRI and image embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        conditioning_dim: Optional[int] = None,
        use_permutations: bool = True,
    ) -> None:
        super().__init__()
        steps: List[FlowStep] = []
        for i in range(num_layers):
            mask = checkerboard_mask(input_dim, even=i % 2 == 0)
            coupling = AffineCoupling(input_dim, hidden_dim, mask, conditioning_dim)
            actnorm = ActNorm(input_dim)
            permute = RandomPermutation(input_dim) if use_permutations else None
            steps.append(FlowStep(coupling=coupling, actnorm=actnorm, permute=permute))
        self.steps = nn.ModuleList(steps)
        self.base_dist = torch.distributions.Normal(0, 1)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        log_det_total = torch.zeros(x.size(0), device=x.device)
        for step in self.steps:
            x, log_det = step(x, cond)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z: Tensor, cond: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        log_det_total = torch.zeros(z.size(0), device=z.device)
        for step in reversed(self.steps):
            z, log_det = step.inverse(z, cond)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        z, log_det = self.forward(x, cond)
        log_pz = self.base_dist.log_prob(z).sum(dim=-1)
        return log_pz + log_det

    def sample(self, num_samples: int, cond: Optional[Tensor] = None, device: Optional[torch.device] = None) -> Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.steps[0].coupling.mask.numel(), device=device)
        x, _ = self.inverse(z, cond)
        return x
