"""
Video Decoder using TorchCodec with NVDEC GPU acceleration.

Supports: H.264, HEVC (H.265), AV1, VP9 codecs
Output: GPU tensors in NCHW format (direct GPU memory, no CPU transfer)
"""

import torch
import torchcodec
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoDecoder:
    """GPU-accelerated video decoder using TorchCodec."""

    def __init__(self, video_path: str, device: str = "cuda:0"):
        """
        Initialize video decoder.

        Args:
            video_path: Path to input video file
            device: PyTorch device string (e.g., "cuda:0")

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If TorchCodec fails to initialize
        """
        self.video_path = Path(video_path)
        self.device = device

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            # Initialize TorchCodec decoder
            # Device string is passed directly; TorchCodec handles CUDA
            self.decoder = torchcodec.decoders._SimpleVideoDecoder(
                str(self.video_path),
                device=device
            )
            logger.info(f"Initialized decoder for {self.video_path.name} on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize decoder: {e}")
            raise RuntimeError(f"TorchCodec initialization failed: {e}")

    def decode_batch(self, start_frame: int, batch_size: int) -> torch.Tensor:
        """
        Decode a batch of frames directly to GPU tensors.

        Args:
            start_frame: Starting frame index (0-based)
            batch_size: Number of consecutive frames to decode

        Returns:
            torch.Tensor: Shape (B, C, H, W) on GPU device, uint8 [0-255]
                         Channels: RGB (not BGR)

        Raises:
            IndexError: If start_frame + batch_size exceeds total frames
        """
        total_frames = len(self.decoder)

        if start_frame >= total_frames:
            raise IndexError(
                f"start_frame {start_frame} >= total frames {total_frames}"
            )

        # Clamp batch_size to available frames
        actual_batch_size = min(batch_size, total_frames - start_frame)

        frames = []
        try:
            for i in range(start_frame, start_frame + actual_batch_size):
                # decoder[i] returns tensor on GPU, shape (C, H, W)
                frame = self.decoder[i]
                frames.append(frame)

            # Stack into batch: (B, C, H, W)
            batch = torch.stack(frames, dim=0)
            return batch

        except Exception as e:
            logger.error(f"Error decoding frames {start_frame}-{start_frame + actual_batch_size}: {e}")
            raise RuntimeError(f"Frame decoding failed: {e}")

    def get_metadata(self) -> dict:
        """
        Get video metadata.

        Returns:
            dict with keys:
                - num_frames: Total frame count
                - width: Frame width in pixels
                - height: Frame height in pixels
                - fps: Frames per second (average)
                - codec: Video codec name (if available)
        """
        metadata = self.decoder.metadata

        return {
            "num_frames": len(self.decoder),
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.average_fps if hasattr(metadata, 'average_fps') else 30.0,
            "codec": str(metadata.codec) if hasattr(metadata, 'codec') else "unknown"
        }

    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self.decoder)

    def __del__(self):
        """Cleanup decoder resources."""
        try:
            # Ensure decoder is properly released
            if hasattr(self, 'decoder'):
                del self.decoder
        except Exception as e:
            logger.warning(f"Error during decoder cleanup: {e}")
