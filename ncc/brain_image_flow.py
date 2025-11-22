"""
Bidirectional fMRI <-> image modeling with conditional normalizing flows.

The design follows RealNVP/Glow-style flows and couples them with strong modality encoders:
- A vision backbone (default: CLIP ViT-B/32 or torchvision ResNet-50 fallback) encodes images.
- A brain encoder (multi-layer perceptron) projects fMRI voxel vectors into a shared embedding.
- Two conditional flows model p(img_embed | fmri_embed) and p(fmri_embed | img_embed).

This module exposes training utilities and interpretability helpers (region importance and
attention heatmaps) to highlight which brain regions are most predictive for a target image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import models

from .flows import ConditionalRealNVP

try:  # Prefer CLIP when available for strong image-text alignment.
    import clip
except Exception:  # pragma: no cover - optional dependency
    clip = None


class BrainEncoder(nn.Module):
    """Project fMRI voxel vectors into a compact embedding suitable for flows."""

    def __init__(self, input_dim: int, hidden_dim: int = 2048, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """Wrapper around CLIP or a torchvision backbone that outputs normalized embeddings."""

    def __init__(self, embed_dim: int = 512, train_backbone: bool = False):
        super().__init__()
        self.train_backbone = train_backbone
        if clip is not None:
            self.backbone, _ = clip.load("ViT-B/32", device="cpu")
            self.proj = nn.Linear(self.backbone.visual.output_dim, embed_dim)
        else:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.proj = nn.Linear(2048, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, images: Tensor) -> Tensor:
        if not self.train_backbone:
            with torch.no_grad():
                feats = self.backbone.encode_image(images) if clip is not None else self.backbone(images)
        else:
            feats = self.backbone.encode_image(images) if clip is not None else self.backbone(images)
        feats = feats if clip is None else feats.float()
        return F.normalize(self.proj(feats), dim=-1)


@dataclass
class ForwardOutputs:
    fmri_to_img_log_prob: Tensor
    img_to_fmri_log_prob: Tensor
    fmri_embeddings: Tensor
    image_embeddings: Tensor


class BrainImageFlowSystem(nn.Module):
    """
    Wrap two conditional flows to jointly model fMRI and image embeddings.

    The architecture encourages cycle consistency by sharing encoders and providing utilities to
    sample in both directions.
    """

    def __init__(
        self,
        fmri_dim: int,
        embed_dim: int = 512,
        flow_hidden: int = 1024,
        flow_layers: int = 6,
        train_image_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.fmri_encoder = BrainEncoder(fmri_dim, embed_dim=embed_dim)
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, train_backbone=train_image_backbone)
        self.flow_img_given_fmri = ConditionalRealNVP(
            input_dim=embed_dim, hidden_dim=flow_hidden, num_layers=flow_layers, conditioning_dim=embed_dim
        )
        self.flow_fmri_given_img = ConditionalRealNVP(
            input_dim=embed_dim, hidden_dim=flow_hidden, num_layers=flow_layers, conditioning_dim=embed_dim
        )

    def forward(self, fmri: Tensor, images: Tensor) -> ForwardOutputs:
        fmri_embed = self.fmri_encoder(fmri)
        img_embed = self.image_encoder(images)
        lp_img = self.flow_img_given_fmri.log_prob(img_embed, cond=fmri_embed)
        lp_fmri = self.flow_fmri_given_img.log_prob(fmri_embed, cond=img_embed)
        return ForwardOutputs(lp_img, lp_fmri, fmri_embed, img_embed)

    @torch.no_grad()
    def sample_images_from_fmri(self, fmri: Tensor, num_samples: int = 1) -> Tensor:
        cond = self.fmri_encoder(fmri)
        return self.flow_img_given_fmri.sample(num_samples, cond=cond)

    @torch.no_grad()
    def sample_fmri_from_images(self, images: Tensor, num_samples: int = 1) -> Tensor:
        cond = self.image_encoder(images)
        return self.flow_fmri_given_img.sample(num_samples, cond=cond)

    def region_importance(self, fmri: Tensor, region_masks: Tensor) -> Tensor:
        """
        Compute region-wise importance scores using gradients of the log-probability.

        Args:
            fmri: [B, V] voxel input.
            region_masks: [R, V] binary masks for anatomical or functional parcels.
        Returns:
            Tensor of shape [B, R] with normalized importance per region.
        """

        fmri = fmri.clone().requires_grad_(True)
        dummy_img = torch.zeros(fmri.size(0), self.flow_img_given_fmri.steps[0].coupling.mask.numel())
        log_prob = self.flow_img_given_fmri.log_prob(dummy_img, cond=self.fmri_encoder(fmri))
        log_prob.sum().backward()
        grads = fmri.grad.abs()  # [B, V]
        scores = grads @ region_masks.T
        return F.normalize(scores, p=1, dim=-1)

    def attention_map(self, fmri: Tensor, images: Tensor) -> Tensor:
        """
        Gradient-based saliency on image embeddings, revealing brain-driven attention hotspots.
        Returns a relevance vector aligned with the image embedding dimensions.
        """

        images = images.clone().requires_grad_(True)
        outputs = self.forward(fmri, images)
        outputs.fmri_to_img_log_prob.sum().backward()
        saliency = images.grad
        return saliency


def nll_loss(outputs: ForwardOutputs, lambda_cycle: float = 0.5) -> Tensor:
    """Combined negative log-likelihood encouraging symmetry between flows."""

    loss_img = -outputs.fmri_to_img_log_prob.mean()
    loss_fmri = -outputs.img_to_fmri_log_prob.mean()
    cycle = F.mse_loss(outputs.fmri_embeddings, outputs.image_embeddings)
    return loss_img + loss_fmri + lambda_cycle * cycle


def configure_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


@dataclass
class TrainState:
    epoch: int
    global_step: int
    metrics: Dict[str, float]


def train_one_epoch(
    model: BrainImageFlowSystem,
    dataloader: Iterable[Tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_cycle: float = 0.5,
) -> TrainState:
    model.train()
    metrics: Dict[str, float] = {"nll": 0.0}
    for step, (fmri, images) in enumerate(dataloader):
        fmri, images = fmri.to(device), images.to(device)
        optimizer.zero_grad()
        outputs = model(fmri, images)
        loss = nll_loss(outputs, lambda_cycle=lambda_cycle)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics["nll"] += loss.item()
    num_batches = step + 1 if isinstance(step, int) else 1
    metrics = {k: v / num_batches for k, v in metrics.items()}
    return TrainState(epoch=0, global_step=num_batches, metrics=metrics)
