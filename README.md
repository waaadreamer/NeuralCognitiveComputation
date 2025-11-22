# NeuralCognitiveComputation

This repository provides a lightweight PyTorch implementation for modeling the joint distribution
between fMRI voxel activity and natural images using conditional normalizing flows. The pipeline
leverages strong image encoders (CLIP ViT-B/32 or ResNet-50 fallback) and RealNVP/Glow-inspired
flows to support bidirectional mapping:

- **fMRI → image embedding**: sample or score images conditioned on recorded brain activity.
- **image → fMRI embedding**: predict likely brain responses conditioned on an image.
- **Interpretability**: gradient-based region importance and attention maps highlight which brain
  parcels contribute to an image prediction and where the brain focuses in the image space.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision pillow
# Optional but recommended for stronger image features
pip install git+https://github.com/openai/CLIP.git
```

## Quick start

```python
import torch
from torch.utils.data import DataLoader
from ncc import BrainImageFlowSystem, FMRIDataset, configure_optimizer, train_one_epoch

# fMRI tensors should be [N, V], image_paths length N
fmri_tensors = torch.randn(8, 1024)
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", "..."]
dataset = FMRIDataset(fmri_tensors=fmri_tensors, image_paths=image_paths)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = BrainImageFlowSystem(fmri_dim=fmri_tensors.size(1), embed_dim=512, flow_layers=6)
optimizer = configure_optimizer(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# One training epoch
train_state = train_one_epoch(model, dataloader, optimizer, device)
print(train_state.metrics)

# Sampling image embeddings conditioned on fMRI
with torch.no_grad():
    fmri_batch, _ = next(iter(dataloader))
    samples = model.sample_images_from_fmri(fmri_batch.to(device), num_samples=4)
    print(samples.shape)  # [4, embed_dim]
```

## Dataset: BOLD5000 fMRI ↔ image pairs

- A large public dataset with ~5k COCO/ImageNet/Scene images and 7T fMRI responses
  from four subjects is available on OpenNeuro (ID: `ds001499`).
- Use the provided helper script to pull a curated subset (stimuli and pycortex
  betas) or the full BIDS tree:

```bash
# install optional dependency
pip install openneuro

# download images + surface betas into ./data/bold5000 (defaults shown)
python scripts/download_bold5000.py --output ./data/bold5000

# fetch the full BIDS archive (large)
python scripts/download_bold5000.py --output ./data/bold5000 --include-all
```

After downloading, map fMRI betas to their corresponding stimulus images using
the BIDS events files or the pycortex derivative metadata. The resulting
flattened beta tensors and aligned image paths can be fed directly into
`ncc.FMRIDataset` for training and evaluation.

## Region importance and attention

- `BrainImageFlowSystem.region_importance(fmri, region_masks)` returns normalized importance per
  region mask (e.g., Desikan atlas parcels) using gradients of the conditional image likelihood.
- `BrainImageFlowSystem.attention_map(fmri, images)` computes gradient saliency on the image
  embeddings to reveal which embedding dimensions are most influenced by the fMRI pattern.

## Notes

- The flow design is RealNVP-style (affine coupling + ActNorm + random permutations) inspired by
  architectures such as Glow, enabling tractable log-likelihoods for both modalities.
- Training combines negative log-likelihoods for both directions with a cycle consistency term that
  aligns fMRI and image embeddings.
- The architecture and objectives mirror the cross-modal normalizing flow approach discussed in
  *Zhang et al., "Cross-Modal Normalizing Flows for 3D Brain Image Synthesis" (IEEE TMI 2022).* 
