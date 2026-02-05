"""
Unit tests for VideoEncoder component.

Tests WebM encoding with alpha channel support.
"""

import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_encoder_initialization():
    """Test encoder initialization."""
    try:
        logger.info("Test: Encoder Initialization")
        from video_encoder import VideoEncoder

        encoder = VideoEncoder("test_output_init.webm", fps=30.0)
        logger.info("  ✅ Initialization successful")

        # Cleanup
        if Path("test_output_init.webm").exists():
            Path("test_output_init.webm").unlink()

        return True

    except Exception as e:
        logger.error(f"  ❌ Initialization failed: {e}")
        return False


def test_encode_batch():
    """Test encoding batch of frames."""
    try:
        logger.info("Test: Encode Batch")
        from video_encoder import VideoEncoder

        # Create dummy RGBA batch (10 frames, 720x1280)
        batch = torch.rand(10, 4, 720, 1280).cuda()

        encoder = VideoEncoder("test_output_batch.webm", fps=30.0)
        encoder.encode_batch(batch)
        encoder.finalize()

        # Check file was created
        output_path = Path("test_output_batch.webm")
        assert output_path.exists(), "Output file not created"
        assert output_path.stat().st_size > 0, "Output file is empty"

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Output size: {file_size_mb:.2f} MB")
        logger.info("  ✅ Batch encoding successful")

        # Cleanup
        output_path.unlink()
        return True

    except Exception as e:
        logger.error(f"  ❌ Batch encoding failed: {e}")
        # Cleanup on error
        if Path("test_output_batch.webm").exists():
            Path("test_output_batch.webm").unlink()
        return False


def test_multiple_batches():
    """Test encoding multiple batches to same file."""
    try:
        logger.info("Test: Multiple Batches")
        from video_encoder import VideoEncoder

        encoder = VideoEncoder("test_output_multi.webm", fps=30.0)

        # Encode 3 batches
        for batch_idx in range(3):
            batch = torch.rand(5, 4, 720, 1280).cuda()
            encoder.encode_batch(batch)

        encoder.finalize()

        output_path = Path("test_output_multi.webm")
        assert output_path.exists(), "Output file not created"

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Total frames: 15, Output size: {file_size_mb:.2f} MB")
        logger.info("  ✅ Multiple batches successful")

        # Cleanup
        output_path.unlink()
        return True

    except Exception as e:
        logger.error(f"  ❌ Multiple batches failed: {e}")
        if Path("test_output_multi.webm").exists():
            Path("test_output_multi.webm").unlink()
        return False


def test_different_resolutions():
    """Test encoding different resolutions."""
    try:
        logger.info("Test: Different Resolutions")
        from video_encoder import VideoEncoder

        resolutions = [(720, 1280), (1080, 1920), (2160, 3840)]

        for idx, (h, w) in enumerate(resolutions):
            output_file = f"test_output_res_{idx}.webm"
            encoder = VideoEncoder(output_file, fps=30.0)

            batch = torch.rand(5, 4, h, w).cuda()
            encoder.encode_batch(batch)
            encoder.finalize()

            output_path = Path(output_file)
            assert output_path.exists()
            output_path.unlink()

        logger.info(f"  Tested resolutions: {resolutions}")
        logger.info("  ✅ Different resolutions successful")
        return True

    except Exception as e:
        logger.error(f"  ❌ Resolution test failed: {e}")
        return False


def test_context_manager():
    """Test context manager usage."""
    try:
        logger.info("Test: Context Manager")
        from video_encoder import VideoEncoder

        with VideoEncoder("test_output_ctx.webm", fps=30.0) as encoder:
            batch = torch.rand(5, 4, 720, 1280).cuda()
            encoder.encode_batch(batch)

        output_path = Path("test_output_ctx.webm")
        assert output_path.exists(), "Output file not created"

        logger.info("  ✅ Context manager successful")

        # Cleanup
        output_path.unlink()
        return True

    except Exception as e:
        logger.error(f"  ❌ Context manager test failed: {e}")
        if Path("test_output_ctx.webm").exists():
            Path("test_output_ctx.webm").unlink()
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("VIDEO ENCODER UNIT TESTS")
    logger.info("=" * 60)
    logger.info("")

    tests = [
        test_encoder_initialization,
        test_encode_batch,
        test_multiple_batches,
        test_different_resolutions,
        test_context_manager
    ]

    results = []
    for test_func in tests:
        results.append(test_func())
        logger.info("")

    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    logger.info("=" * 60)

    exit(0 if passed == total else 1)
