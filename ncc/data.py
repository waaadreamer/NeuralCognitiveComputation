"""Dataset helpers for paired fMRI-image observations."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class FMRIDataset(Dataset):
    """
    Simple paired dataset for fMRI voxel vectors and natural images.

    Args:
        fmri_tensors: Tensor of shape [N, V] or path to a .pt tensor file.
        image_paths: Sequence of image file paths aligned with fmri_tensors.
        image_transform: Optional torchvision transform; defaults to ImageNet normalization.
    """

    def __init__(
        self,
        fmri_tensors: Tensor | str | Path,
        image_paths: Sequence[str | Path],
        image_transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ) -> None:
        super().__init__()
        if isinstance(fmri_tensors, (str, Path)):
            fmri_tensors = torch.load(fmri_tensors)
        self.fmri_tensors = fmri_tensors.float()
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = image_transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if len(self.fmri_tensors) != len(self.image_paths):
            raise ValueError("Number of fMRI samples and images must match")

    def __len__(self) -> int:
        return len(self.fmri_tensors)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        fmri = self.fmri_tensors[idx]
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return fmri, self.transform(image)
