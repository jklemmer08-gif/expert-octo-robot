"""
Video Encoder using imageio with VP9 codec for WebM output.

Supports alpha channel (RGBA) encoding to WebM with yuva420p pixel format.
Input: GPU tensors (NCHW format)
Output: WebM file with alpha channel support
"""

import torch
import imageio
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoEncoder:
    """WebM video encoder with alpha channel support."""

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        codec: str = "libvpx-vp9",
        pixelformat: str = "yuva420p"
    ):
        """
        Initialize video encoder.

        Args:
            output_path: Path to output WebM file
            fps: Frames per second for output video
            codec: Video codec ("libvpx-vp9" for VP9, supports alpha)
            pixelformat: Pixel format ("yuva420p" includes alpha channel)

        Raises:
            ValueError: If codec doesn't support alpha or invalid parameters
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.pixelformat = pixelformat

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate codec/pixelformat combo
        if "yuva" not in pixelformat and codec == "libvpx-vp9":
            logger.warning(
                f"pixelformat '{pixelformat}' may not preserve alpha with VP9. "
                f"Consider 'yuva420p' for transparent output."
            )

        logger.info(
            f"Encoder configured: {codec} @ {fps} fps -> {output_path.name}"
        )

        self.writer = None
        self.frame_count = 0

    def _open_writer(self, height: int, width: int) -> None:
        """
        Open imageio video writer with configured parameters.

        Args:
            height: Frame height
            width: Frame width
        """
        try:
            self.writer = imageio.get_writer(
                str(self.output_path),
                codec=self.codec,
                pixelformat=self.pixelformat,
                fps=self.fps
            )
            logger.debug(f"Writer opened: {width}x{height} @ {self.fps} fps")
        except Exception as e:
            logger.error(f"Failed to open writer: {e}")
            raise RuntimeError(f"Writer initialization failed: {e}")

    def encode_batch(self, batch_rgba: torch.Tensor) -> None:
        """
        Encode batch of RGBA frames to WebM.

        Args:
            batch_rgba: Tensor of shape (B, 4, H, W)
                       - dtype: float32
                       - values: [0, 1] range
                       - channels: RGBA order
                       - on GPU (will be moved to CPU)

        Raises:
            RuntimeError: If encoding fails
            ValueError: If batch format is invalid
        """
        try:
            # Validate input
            if not isinstance(batch_rgba, torch.Tensor):
                raise ValueError("batch_rgba must be a PyTorch tensor")

            if batch_rgba.ndim != 4:
                raise ValueError(f"Expected 4D tensor, got shape {batch_rgba.shape}")

            batch_size, channels, height, width = batch_rgba.shape

            if channels != 4:
                raise ValueError(f"Expected 4 channels (RGBA), got {channels}")

            # Initialize writer on first batch
            if self.writer is None:
                self._open_writer(height, width)

            # Ensure values are in [0, 1] range and clamp
            batch_rgba = batch_rgba.clamp(0, 1)

            # Convert to uint8 [0, 255]
            batch_uint8 = (batch_rgba * 255).byte()

            # Move to CPU and convert to numpy
            batch_np = batch_uint8.cpu().numpy()

            # Transpose from (B, C, H, W) to (B, H, W, C)
            batch_np = batch_np.transpose(0, 2, 3, 1)

            # Write each frame
            for frame_idx, frame in enumerate(batch_np):
                # imageio expects RGB order in last dimension
                # Input is RGBA, output should be RGBA
                # No channel swapping needed for RGBA
                self.writer.append_data(frame)
                self.frame_count += 1

            logger.debug(f"Encoded {batch_size} frames ({self.frame_count} total)")

        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            if self.writer is not None:
                try:
                    self.writer.close()
                except Exception:
                    pass
            raise RuntimeError(f"Batch encoding failed: {e}")

    def finalize(self) -> None:
        """
        Close the video writer and finalize the output file.

        Should be called after all batches have been encoded.

        Raises:
            RuntimeError: If finalization fails
        """
        try:
            if self.writer is not None:
                self.writer.close()
                self.writer = None

                file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"Video finalized: {self.frame_count} frames, "
                    f"{file_size_mb:.1f} MB -> {self.output_path.name}"
                )

        except Exception as e:
            logger.error(f"Error finalizing video: {e}")
            raise RuntimeError(f"Finalization failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure writer is closed."""
        self.finalize()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception as e:
            logger.warning(f"Error during encoder cleanup: {e}")
