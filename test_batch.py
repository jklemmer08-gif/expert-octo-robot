"""
Batch processing tests.

Tests dual GPU parallel processing with multiple videos.
"""

import logging
from pathlib import Path
from batch_processor import BatchProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dual_gpu_processing():
    """Test processing multiple videos across 2 GPUs."""
    try:
        logger.info("Test: Dual GPU Batch Processing")

        # Create test videos list
        videos = [Path("test_av1.mp4"), Path("test_h264.mp4")]

        # Verify test videos exist
        for video in videos:
            if not video.exists():
                logger.warning(f"  Test video not found: {video}")
                return False

        processor = BatchProcessor(model="resnet50")

        results = processor.process_batch(
            videos=videos,
            output_dir=Path("output_batch_test"),
            num_gpus=2
        )

        # Check results
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = sum(1 for r in results if r.get("status") == "failed")

        logger.info(f"  Successful: {successful}, Failed: {failed}")

        for result in results:
            if result.get("status") == "success":
                logger.info(f"    ✅ {result['video']} (GPU {result['gpu']})")
            else:
                logger.info(f"    ❌ {result['video']}: {result.get('error')}")

        if successful == len(videos):
            logger.info("  ✅ Dual GPU batch processing successful")

            # Cleanup
            import shutil
            if Path("output_batch_test").exists():
                shutil.rmtree("output_batch_test")

            return True
        else:
            logger.error(f"  ❌ Only {successful}/{len(videos)} videos succeeded")
            return False

    except Exception as e:
        logger.error(f"  ❌ Batch processing test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING TESTS")
    logger.info("=" * 60)
    logger.info("")

    results = []
    results.append(test_dual_gpu_processing())

    logger.info("")
    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    logger.info("=" * 60)

    exit(0 if passed == total else 1)
