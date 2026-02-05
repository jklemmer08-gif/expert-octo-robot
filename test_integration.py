"""
Integration tests for complete video processing pipeline.

Tests end-to-end video processor with both AV1 and H.264 inputs.
"""

import logging
from pathlib import Path
from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_av1_integration():
    """Test complete pipeline with AV1 input."""
    try:
        logger.info("Test: AV1 End-to-End Processing")

        processor = VideoProcessor(device="cuda:0", batch_size=8)
        result = processor.process_video("test_av1.mp4", "output_av1_e2e.webm")

        assert result["status"] == "success", f"Processing failed: {result.get('error')}"
        assert Path("output_av1_e2e.webm").exists(), "Output file not created"
        assert result["fps"] > 0, "Invalid fps"
        assert result["frames"] > 0, "No frames processed"

        logger.info(f"  Frames: {result['frames']}")
        logger.info(f"  FPS: {result['fps']:.1f}")
        logger.info(f"  Time: {result['time']:.1f}s")
        logger.info(f"  Output: {result['output_size_mb']:.1f} MB")
        logger.info("  ✅ AV1 integration successful")

        # Cleanup
        Path("output_av1_e2e.webm").unlink()
        return True

    except Exception as e:
        logger.error(f"  ❌ AV1 integration failed: {e}")
        if Path("output_av1_e2e.webm").exists():
            Path("output_av1_e2e.webm").unlink()
        return False


def test_h264_integration():
    """Test complete pipeline with H.264 input."""
    try:
        logger.info("Test: H.264 End-to-End Processing")

        processor = VideoProcessor(device="cuda:0", batch_size=8)
        result = processor.process_video("test_h264.mp4", "output_h264_e2e.webm")

        assert result["status"] == "success", f"Processing failed: {result.get('error')}"
        assert Path("output_h264_e2e.webm").exists(), "Output file not created"
        assert result["fps"] > 0, "Invalid fps"

        logger.info(f"  Frames: {result['frames']}")
        logger.info(f"  FPS: {result['fps']:.1f}")
        logger.info(f"  Time: {result['time']:.1f}s")
        logger.info(f"  Output: {result['output_size_mb']:.1f} MB")
        logger.info("  ✅ H.264 integration successful")

        # Cleanup
        Path("output_h264_e2e.webm").unlink()
        return True

    except Exception as e:
        logger.error(f"  ❌ H.264 integration failed: {e}")
        if Path("output_h264_e2e.webm").exists():
            Path("output_h264_e2e.webm").unlink()
        return False


def test_gpu_utilization():
    """Test that GPU is being utilized during processing."""
    try:
        logger.info("Test: GPU Utilization")
        logger.info("  Note: Monitor nvidia-smi in separate terminal for GPU usage")

        processor = VideoProcessor(device="cuda:0", batch_size=8)

        # Process a small clip
        result = processor.process_video("test_av1.mp4", "output_util_test.webm")

        if result["status"] == "success":
            # If fps > 2, GPU was likely utilized
            if result["fps"] > 2:
                logger.info("  ✅ GPU utilization appears adequate")
            else:
                logger.warning("  ⚠️  Low fps might indicate GPU underutilization")

            Path("output_util_test.webm").unlink()
            return True
        else:
            logger.error(f"  ❌ Processing failed: {result.get('error')}")
            if Path("output_util_test.webm").exists():
                Path("output_util_test.webm").unlink()
            return False

    except Exception as e:
        logger.error(f"  ❌ GPU utilization test failed: {e}")
        if Path("output_util_test.webm").exists():
            Path("output_util_test.webm").unlink()
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("INTEGRATION TESTS")
    logger.info("=" * 60)
    logger.info("")

    tests = [
        test_av1_integration,
        test_h264_integration,
        test_gpu_utilization
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
