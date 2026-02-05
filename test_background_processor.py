"""
Unit tests for BackgroundProcessor component.

Tests RVM-based background removal on GPU.
"""

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_processor_initialization():
    """Test processor initialization."""
    try:
        logger.info("Test: Processor Initialization")
        from background_processor import BackgroundProcessor

        processor = BackgroundProcessor(device="cuda:0", batch_size=8)
        logger.info("  ✅ Initialization successful")
        return True

    except Exception as e:
        logger.error(f"  ❌ Initialization failed: {e}")
        return False


def test_batch_processing():
    """Test processing batch through RVM."""
    try:
        logger.info("Test: Batch Processing")
        from background_processor import BackgroundProcessor

        processor = BackgroundProcessor(device="cuda:0", batch_size=8)

        # Create dummy batch (8, 3, 1080, 1920) RGB values [0, 255]
        batch = torch.randint(0, 256, (8, 3, 1080, 1920), dtype=torch.uint8).cuda()

        # Process batch
        rgba = processor.process_batch(batch)

        # Validate output
        assert rgba.shape == (8, 4, 1080, 1920), f"Shape mismatch: {rgba.shape}"
        assert rgba.dtype == torch.float32, f"Expected float32, got {rgba.dtype}"
        assert rgba.device.type == 'cuda', "Output not on GPU"
        assert rgba.min() >= 0 and rgba.max() <= 1, "Values out of range"

        logger.info(f"  Output shape: {rgba.shape}")
        logger.info(f"  Output range: [{rgba.min():.3f}, {rgba.max():.3f}]")
        logger.info("  ✅ Batch processing successful")
        return True

    except Exception as e:
        logger.error(f"  ❌ Batch processing failed: {e}")
        return False


def test_recurrent_states():
    """Test recurrent state management."""
    try:
        logger.info("Test: Recurrent States")
        from background_processor import BackgroundProcessor

        processor = BackgroundProcessor(device="cuda:0", batch_size=8)

        # Process a batch to initialize states
        batch = torch.randint(0, 256, (4, 3, 1080, 1920), dtype=torch.uint8).cuda()
        rgba1 = processor.process_batch(batch)

        # Check states are initialized
        state_shapes = processor.get_recurrent_state_size()
        assert state_shapes[0] is not None, "States not initialized"
        logger.info(f"  State shapes: {state_shapes}")

        # Reset states
        processor.reset_recurrent_states()
        state_shapes_reset = processor.get_recurrent_state_size()
        assert state_shapes_reset[0] is None, "States not reset"

        logger.info("  ✅ Recurrent state management successful")
        return True

    except Exception as e:
        logger.error(f"  ❌ Recurrent state test failed: {e}")
        return False


def test_different_batch_sizes():
    """Test processing different batch sizes."""
    try:
        logger.info("Test: Different Batch Sizes")
        from background_processor import BackgroundProcessor

        processor = BackgroundProcessor(device="cuda:0")

        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            batch = torch.randint(0, 256, (batch_size, 3, 720, 1280), dtype=torch.uint8).cuda()
            rgba = processor.process_batch(batch)

            assert rgba.shape[0] == batch_size, f"Batch size mismatch for size {batch_size}"

        logger.info(f"  Tested batch sizes: {batch_sizes}")
        logger.info("  ✅ Different batch sizes successful")
        return True

    except Exception as e:
        logger.error(f"  ❌ Batch size test failed: {e}")
        return False


def test_different_resolutions():
    """Test processing different resolutions."""
    try:
        logger.info("Test: Different Resolutions")
        from background_processor import BackgroundProcessor

        processor = BackgroundProcessor(device="cuda:0")

        resolutions = [(720, 1280), (1080, 1920), (2160, 3840)]
        for h, w in resolutions:
            batch = torch.randint(0, 256, (4, 3, h, w), dtype=torch.uint8).cuda()
            rgba = processor.process_batch(batch)

            assert rgba.shape == (4, 4, h, w), f"Shape mismatch for {h}x{w}"

        logger.info(f"  Tested resolutions: {resolutions}")
        logger.info("  ✅ Different resolutions successful")
        return True

    except Exception as e:
        logger.error(f"  ❌ Resolution test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BACKGROUND PROCESSOR UNIT TESTS")
    logger.info("=" * 60)
    logger.info("")

    tests = [
        test_processor_initialization,
        test_batch_processing,
        test_recurrent_states,
        test_different_batch_sizes,
        test_different_resolutions
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
