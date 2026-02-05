"""
Background Removal Processor using Robust Video Matting (RVM).

Processes video frames through ResNet50-based RVM model to extract
foreground with alpha channel on GPU, maintaining recurrent state
across batch sequences.

Input: RGB frames [0, 255]
Output: RGBA frames [0, 1] with alpha channel
"""

import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class BackgroundProcessor:
    """GPU-accelerated background removal using RVM model."""

    def __init__(
        self,
        model: str = "resnet50",
        device: str = "cuda:0",
        batch_size: int = 8,
        downsample_ratio: float = 0.5
    ):
        """
        Initialize background processor with RVM model.

        Args:
            model: RVM model variant ("resnet50" or other variants)
            device: PyTorch device string (e.g., "cuda:0")
            batch_size: Expected batch size (for memory allocation)
            downsample_ratio: Processing resolution ratio (0.5 = half res for speed)

        Raises:
            RuntimeError: If model loading fails
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.downsample_ratio = downsample_ratio

        try:
            # Load RVM model from PyTorch Hub
            logger.info(f"Loading RVM {model} model...")
            self.model = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                model,
                pretrained=True
            ).to(self.device).eval()

            logger.info(f"RVM model loaded on {device}")

            # Recurrent states: [r1, r2, r3, r4]
            # Maintained across batches to ensure temporal consistency
            self.rec = [None] * 4

        except Exception as e:
            logger.error(f"Failed to load RVM model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def process_batch(self, batch_rgb: torch.Tensor) -> torch.Tensor:
        """
        Process batch of RGB frames through RVM.

        Args:
            batch_rgb: Tensor of shape (B, C, H, W)
                      - dtype: uint8 or float32
                      - values: [0, 255] for uint8, [0, 1] for float32
                      - channels: RGB order
                      - already on GPU device

        Returns:
            torch.Tensor: Shape (B, 4, H, W)
                         - dtype: float32
                         - values: [0, 1] range
                         - channels: RGBA (alpha in channel 3)
                         - on same GPU device as input

        Raises:
            RuntimeError: If processing fails
            ValueError: If batch format is invalid
        """
        try:
            # Validate input
            if not isinstance(batch_rgb, torch.Tensor):
                raise ValueError("batch_rgb must be a PyTorch tensor")

            if batch_rgb.ndim != 4:
                raise ValueError(f"Expected 4D tensor, got shape {batch_rgb.shape}")

            batch_size, channels, height, width = batch_rgb.shape

            if channels != 3:
                raise ValueError(f"Expected 3 channels (RGB), got {channels}")

            # Ensure tensor is on correct device
            batch_rgb = batch_rgb.to(self.device)

            # Normalize to [0, 1] if input is uint8
            if batch_rgb.dtype == torch.uint8:
                batch_rgb = batch_rgb.float() / 255.0
            elif batch_rgb.dtype == torch.float32:
                # Ensure values are in [0, 1]
                batch_rgb = batch_rgb.clamp(0, 1)
            else:
                raise ValueError(f"Unsupported dtype: {batch_rgb.dtype}")

            # Process through RVM without gradient computation
            with torch.no_grad():
                # RVM expects 3-channel RGB input
                # Returns: (fgr, pha, *new_rec)
                # fgr: foreground (B, 3, H, W)
                # pha: alpha matte (B, 1, H, W)
                # rec: new recurrent states
                fgr, pha, *self.rec = self.model(
                    batch_rgb,
                    *self.rec,
                    downsample_ratio=self.downsample_ratio
                )

                # Combine foreground and alpha into RGBA
                # fgr: (B, 3, H, W)
                # pha: (B, 1, H, W)
                # output: (B, 4, H, W)
                rgba = torch.cat([fgr, pha], dim=1)

            logger.debug(
                f"Processed batch: input {batch_rgb.shape} -> output {rgba.shape}"
            )

            return rgba

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise RuntimeError(f"Batch processing failed: {e}")

    def reset_recurrent_states(self) -> None:
        """
        Reset recurrent states.

        Should be called between videos to prevent temporal continuity
        across unrelated video sequences.
        """
        self.rec = [None] * 4
        logger.debug("Recurrent states reset")

    def get_recurrent_state_size(self) -> Tuple[int, ...]:
        """
        Get shape of recurrent states.

        Returns:
            Tuple of state tensor shapes
        """
        if self.rec[0] is not None:
            return tuple(s.shape if s is not None else None for s in self.rec)
        return (None, None, None, None)

    def __del__(self):
        """Cleanup model resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'rec'):
                self.rec = [None] * 4
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
