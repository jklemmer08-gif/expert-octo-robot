"""
Integrated Video Processor combining decoder, background removal, and encoder.

Orchestrates end-to-end processing:
1. Decode video frames using TorchCodec (GPU accelerated)
2. Process through RVM for background removal (maintaining recurrent state)
3. Encode result to WebM with alpha channel

Includes monitoring, progress tracking, and error handling.
"""

import torch
import time
import logging
from pathlib import Path
from typing import Dict, Any
from video_decoder import VideoDecoder
from background_processor import BackgroundProcessor
from video_encoder import VideoEncoder

logger = logging.getLogger(__name__)


class VideoProcessor:
    """End-to-end GPU-accelerated video processor."""

    def __init__(
        self,
        model: str = "resnet50",
        device: str = "cuda:0",
        batch_size: int = 8
    ):
        """
        Initialize video processor.

        Args:
            model: RVM model variant
            device: PyTorch device string
            batch_size: Batch size for processing
        """
        self.device = device
        self.batch_size = batch_size
        self.bg_processor = BackgroundProcessor(
            model=model,
            device=device,
            batch_size=batch_size
        )

        logger.info(f"VideoProcessor initialized on {device} with batch_size={batch_size}")

    def _auto_adjust_batch_size(self, metadata: Dict[str, Any]) -> int:
        """
        Auto-adjust batch size based on video resolution.

        Larger resolutions require smaller batches to fit in GPU memory.

        Args:
            metadata: Video metadata dict with 'width' and 'height'

        Returns:
            Recommended batch size
        """
        pixels = metadata['width'] * metadata['height']

        if pixels >= 7680 * 4320:  # 8K
            batch_size = 4
        elif pixels >= 3840 * 2160:  # 4K
            batch_size = 8
        else:  # HD or lower
            batch_size = 16

        if batch_size != self.batch_size:
            logger.info(f"Adjusted batch size: {self.batch_size} -> {batch_size} (resolution {metadata['width']}x{metadata['height']})")
            self.batch_size = batch_size

        return batch_size

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Process single video end-to-end.

        Args:
            input_path: Path to input video
            output_path: Path to output WebM
            progress_callback: Optional callback(processed_frames, total_frames)

        Returns:
            Dict with processing results:
                - status: "success" or "failed"
                - frames: total frames processed
                - time: processing time in seconds
                - fps: average fps during processing
                - resolution: (width, height)
                - input_file: input filename
                - output_file: output filename
                - output_size_mb: output file size in MB

        Raises:
            RuntimeError: If processing fails at any stage
        """
        start_time = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)

        try:
            logger.info(f"Starting processing: {input_path.name}")

            # ===== STAGE 1: Initialize decoder =====
            logger.info("Stage 1: Initializing decoder...")
            decoder = VideoDecoder(str(input_path), device=self.device)
            metadata = decoder.get_metadata()

            logger.info(f"  Codec: {metadata['codec']}")
            logger.info(f"  Resolution: {metadata['width']}x{metadata['height']}")
            logger.info(f"  FPS: {metadata['fps']:.2f}")
            logger.info(f"  Total frames: {metadata['num_frames']}")

            # ===== STAGE 2: Auto-adjust batch size =====
            logger.info("Stage 2: Configuring batch processing...")
            batch_size = self._auto_adjust_batch_size(metadata)
            logger.info(f"  Batch size: {batch_size}")

            # ===== STAGE 3: Reset processor state =====
            logger.info("Stage 3: Initializing processor...")
            self.bg_processor.reset_recurrent_states()

            # ===== STAGE 4: Process in batches =====
            logger.info("Stage 4: Processing frames...")
            encoder = VideoEncoder(str(output_path), fps=metadata['fps'])

            num_frames = metadata['num_frames']
            processed_frames = 0

            for start_idx in range(0, num_frames, batch_size):
                end_idx = min(start_idx + batch_size, num_frames)
                current_batch_size = end_idx - start_idx

                # Decode batch
                batch_rgb = decoder.decode_batch(start_idx, current_batch_size)

                # Ensure RGB range [0, 255]
                if batch_rgb.dtype == torch.uint8:
                    pass  # Already correct
                else:
                    batch_rgb = batch_rgb.clamp(0, 255).byte()

                # Process through RVM
                batch_rgba = self.bg_processor.process_batch(batch_rgb)

                # Encode batch
                encoder.encode_batch(batch_rgba)

                processed_frames = end_idx

                # Progress reporting
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed if elapsed > 0 else 0

                if progress_callback:
                    progress_callback(processed_frames, num_frames)

                # Log every 10 batches
                batch_num = (end_idx // batch_size)
                if batch_num % 10 == 0 or end_idx == num_frames:
                    logger.info(
                        f"  Progress: {processed_frames}/{num_frames} frames "
                        f"({fps:.1f} fps, {elapsed:.1f}s)"
                    )

            # ===== STAGE 5: Finalize =====
            logger.info("Stage 5: Finalizing video...")
            encoder.finalize()

            total_time = time.time() - start_time
            avg_fps = num_frames / total_time if total_time > 0 else 0

            output_size_mb = output_path.stat().st_size / (1024 * 1024)

            logger.info(f"✅ Processing complete in {total_time:.1f}s ({avg_fps:.1f} fps)")
            logger.info(f"   Output: {output_size_mb:.1f} MB")

            return {
                "status": "success",
                "frames": num_frames,
                "time": total_time,
                "fps": avg_fps,
                "resolution": (metadata['width'], metadata['height']),
                "input_file": input_path.name,
                "output_file": output_path.name,
                "output_size_mb": output_size_mb
            }

        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "input_file": input_path.name,
                "output_file": output_path.name
            }

    def __del__(self):
        """Cleanup."""
        try:
            if hasattr(self, 'bg_processor'):
                del self.bg_processor
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
