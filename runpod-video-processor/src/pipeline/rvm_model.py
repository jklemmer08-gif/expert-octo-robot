"""RVM (Robust Video Matting) TorchScript model loading and batch inference."""

import logging
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None  # Tests can mock this

from src.config import AVAILABLE_RVM_MODELS, RVM_MODEL_DIR

logger = logging.getLogger(__name__)


class RVMError(Exception):
    """Raised when RVM processing fails."""
    pass


class RVMProcessor:
    """Wraps a TorchScript RVM model for batch background removal.

    Maintains recurrent hidden states across calls for temporal consistency.
    """

    def __init__(self, model_name: str = "resnet50", device: str = "cuda:0",
                 downsample_ratio: float = 0.5):
        model_info = AVAILABLE_RVM_MODELS.get(model_name)
        if not model_info:
            raise ValueError(
                f"Unknown RVM model: {model_name}. "
                f"Available: {list(AVAILABLE_RVM_MODELS.keys())}"
            )

        model_path = RVM_MODEL_DIR / model_info["file"]
        if not model_path.exists():
            raise FileNotFoundError(
                f"RVM model file not found: {model_path}. "
                "Rebuild the Docker image to download models."
            )

        self.device = device
        self.downsample_ratio = downsample_ratio
        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()
        self.rec: List[Optional[torch.Tensor]] = [None] * 4

        logger.info(
            "Loaded RVM model %s (downsample=%.2f, device=%s)",
            model_name, downsample_ratio, device,
        )

    def process_batch(self, batch_rgb: "torch.Tensor") -> "torch.Tensor":
        """Process a batch of RGB frames through RVM.

        Args:
            batch_rgb: (B, 3, H, W) float32 tensor in [0, 1] on GPU.

        Returns:
            (B, 4, H, W) float32 RGBA tensor in [0, 1].
        """
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(
                batch_rgb, *self.rec, self.downsample_ratio
            )
        rgba = torch.cat([fgr * pha, pha], dim=1).clamp(0, 1)
        return rgba

    def reset(self):
        """Reset recurrent states (call between unrelated videos)."""
        self.rec = [None] * 4

    def get_recurrent_states(self) -> List[Optional["torch.Tensor"]]:
        """Clone current recurrent states for cross-segment persistence."""
        return [r.clone() if r is not None else None for r in self.rec]

    def set_recurrent_states(self, states: List[Optional["torch.Tensor"]]):
        """Restore previously saved recurrent states."""
        self.rec = [s.clone() if s is not None else None for s in states]
