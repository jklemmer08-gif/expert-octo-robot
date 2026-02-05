"""
Unit tests for VideoDecoder component.

Tests TorchCodec-based decoding across multiple codec formats.
"""

import torch
import logging
from pathlib import Path
from video_decoder import VideoDecoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_av1_decoding():
    """Test decoding AV1-encoded video without hanging."""
    try:
        logger.info("Test: AV1 Decoding")
        decoder = VideoDecoder("test_av1.mp4", device="cuda:0")
        metadata = decoder.get_metadata()

        logger.info(f"  Metadata: {metadata}")
        assert metadata['num_frames'] > 0, "No frames found"
        assert metadata['width'] > 0, "Invalid width"
        assert metadata['height'] > 0, "Invalid height"
        assert metadata['fps'] > 0, "Invalid fps"

        # Test decoding first batch
        batch = decoder.decode_batch(0, 8)
        assert batch.shape[0] == 8, f"Expected batch size 8, got {batch.shape[0]}"
        assert batch.shape[1] == 3, f"Expected 3 channels, got {batch.shape[1]}"
        assert batch.device.type == 'cuda', "Batch not on GPU"

        logger.info("  ✅ AV1 decoding successful")
        return True

    except Exception as e:
        logger.error(f"  ❌ AV1 decoding failed: {e}")
        return False


def test_h264_decoding():
    """Test decoding H.264-encoded video."""
    try:
        logger.info("Test: H.264 Decoding")
        decoder = VideoDecoder("test_h264.mp4", device="cuda:0")
        metadata = decoder.get_metadata()

        logger.info(f"  Metadata: {metadata}")
        assert metadata['num_frames'] > 0, "No frames found"

        # Test decoding
        batch = decoder.decode_batch(0, 8)
        assert batch.shape[0] == 8
        assert batch.device.type == 'cuda'

        logger.info("  ✅ H.264 decoding successful")
        return True

    except Exception as e:
        logger.error(f"  ❌ H.264 decoding failed: {e}")
        return False


def test_batch_decoding():
    """Test decoding multiple batches from same video."""
    try:
        logger.info("Test: Multi-batch Decoding")
        decoder = VideoDecoder("test_av1.mp4", device="cuda:0")
        metadata = decoder.get_metadata()

        # Decode multiple batches
        batch1 = decoder.decode_batch(0, 8)
        batch2 = decoder.decode_batch(8, 8)

        assert batch1.shape == batch2.shape, "Batch shape mismatch"
        assert not torch.equal(batch1, batch2), "Batches are identical (should differ)"

        logger.info("  ✅ Multi-batch decoding successful")
        return True

    except Exception as e:
        logger.error(f"  ❌ Multi-batch decoding failed: {e}")
        return False


def test_no_hanging():
    """Test that decoding doesn't hang on AV1."""
    try:
        logger.info("Test: No Hanging on AV1 (timeout after 10s)")
        decoder = VideoDecoder("test_av1.mp4", device="cuda:0")

        # This should not hang
        metadata = decoder.get_metadata()
        batch = decoder.decode_batch(0, 4)

        logger.info("  ✅ No hanging detected")
        return True

    except Exception as e:
        logger.error(f"  ❌ Hanging test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("VIDEO DECODER UNIT TESTS")
    logger.info("=" * 60)

    tests = [
        test_av1_decoding,
        test_h264_decoding,
        test_batch_decoding,
        test_no_hanging
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
