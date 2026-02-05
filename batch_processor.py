"""
Batch Processor with Dual GPU Parallelization.

Processes multiple videos in parallel across available GPUs using multiprocessing.
Each GPU processes its own video queue sequentially, maximizing throughput.

Architecture:
  - Main process: Orchestrates job distribution
  - Worker processes: One per GPU, processes videos sequentially
  - IPC: Queue-based communication for progress and results
"""

import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any
import logging
import time
from video_processor import VideoProcessor

# Configure multiprocessing
mp.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)


def process_video_worker(
    video_path: Path,
    output_dir: Path,
    gpu_id: int,
    model: str = "resnet50",
    result_queue: mp.Queue = None
) -> Dict[str, Any]:
    """
    Worker function for parallel video processing on a single GPU.

    Runs in a separate process. Sets GPU context and processes videos sequentially.

    Args:
        video_path: Path to input video
        output_dir: Output directory for results
        gpu_id: GPU index to use
        model: RVM model variant
        result_queue: Optional queue for returning results to parent

    Returns:
        Dict with processing results
    """
    device = f"cuda:{gpu_id}"

    try:
        # Set GPU context for this process
        torch.cuda.set_device(gpu_id)

        logger.info(f"[GPU {gpu_id}] Starting worker process for {video_path.name}")

        # Initialize processor
        processor = VideoProcessor(model=model, device=device, batch_size=8)

        # Generate output path
        output_path = output_dir / f"{video_path.stem}_transparent.webm"

        # Process video
        logger.info(f"[GPU {gpu_id}] Processing {video_path.name}...")
        result = processor.process_video(str(video_path), str(output_path))

        # Add metadata
        result["gpu"] = gpu_id
        result["video"] = video_path.name

        logger.info(f"[GPU {gpu_id}] ✅ {video_path.name}: {result['status']}")

        # Return via queue if provided
        if result_queue is not None:
            result_queue.put(result)

        return result

    except Exception as e:
        logger.error(f"[GPU {gpu_id}] ❌ Error processing {video_path.name}: {e}")

        result = {
            "status": "failed",
            "error": str(e),
            "gpu": gpu_id,
            "video": video_path.name if video_path else "unknown"
        }

        if result_queue is not None:
            result_queue.put(result)

        return result


class BatchProcessor:
    """Multi-GPU batch video processor."""

    def __init__(self, model: str = "resnet50"):
        """
        Initialize batch processor.

        Args:
            model: RVM model variant to use
        """
        self.model = model
        logger.info(f"BatchProcessor initialized with model: {model}")

    def process_batch(
        self,
        videos: List[Path],
        output_dir: Path,
        num_gpus: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Process multiple videos in parallel across GPUs.

        Distributes videos to GPU workers in round-robin fashion.
        Each GPU processes its assigned videos sequentially.

        Args:
            videos: List of input video paths
            output_dir: Output directory for results
            num_gpus: Number of GPUs to utilize

        Returns:
            List of result dicts, one per video

        Raises:
            RuntimeError: If processing encounters fatal errors
        """
        videos = [Path(v) for v in videos]
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Verify input videos exist
        for video in videos:
            if not video.exists():
                raise FileNotFoundError(f"Video not found: {video}")

        logger.info(f"Starting batch processing: {len(videos)} videos across {num_gpus} GPUs")
        logger.info(f"Output directory: {output_dir}")

        start_time = time.time()
        results = []

        # Distribute videos to GPU workers
        gpu_queues = {gpu_id: [] for gpu_id in range(num_gpus)}

        for idx, video_path in enumerate(videos):
            gpu_id = idx % num_gpus
            gpu_queues[gpu_id].append(video_path)

        # Log distribution
        for gpu_id in range(num_gpus):
            count = len(gpu_queues[gpu_id])
            if count > 0:
                video_names = [v.name for v in gpu_queues[gpu_id]]
                logger.info(f"GPU {gpu_id}: {count} video(s) - {video_names}")

        # Create result queue for communication
        result_queue = mp.Queue()

        # Spawn worker processes
        processes = []

        for gpu_id in range(num_gpus):
            if len(gpu_queues[gpu_id]) == 0:
                continue

            for video_path in gpu_queues[gpu_id]:
                # Create process for this video
                p = mp.Process(
                    target=process_video_worker,
                    args=(video_path, output_dir, gpu_id, self.model, result_queue)
                )
                p.start()
                processes.append(p)

        logger.info(f"Spawned {len(processes)} worker processes")

        # Collect results as processes complete
        for _ in range(len(processes)):
            try:
                result = result_queue.get(timeout=7200)  # 2 hour timeout per video
                results.append(result)
            except mp.TimeoutError:
                logger.error("Worker process timeout - video processing took too long")
                results.append({
                    "status": "failed",
                    "error": "Processing timeout (>2 hours)"
                })

        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=60)
            if p.is_alive():
                logger.warning("Force terminating process...")
                p.terminate()

        # Generate summary
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = sum(1 for r in results if r.get("status") == "failed")

        logger.info("=" * 60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total videos: {len(videos)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")

        if successful > 0:
            total_frames = sum(r.get("frames", 0) for r in results if r.get("status") == "success")
            logger.info(f"Total frames processed: {total_frames}")
            logger.info(f"Average fps: {total_frames / total_time:.1f}")

        logger.info("=" * 60)

        # Log individual results
        logger.info("Individual Results:")
        for result in results:
            if result.get("status") == "success":
                logger.info(
                    f"  ✅ {result['video']} (GPU {result['gpu']}): "
                    f"{result['fps']:.1f} fps, {result['time']:.1f}s, "
                    f"{result['output_size_mb']:.1f} MB"
                )
            else:
                logger.info(
                    f"  ❌ {result.get('video', 'unknown')} (GPU {result.get('gpu', '?')}): "
                    f"{result.get('error', 'Unknown error')}"
                )

        return results


def main():
    """
    Example usage of BatchProcessor.

    Configure input videos and output directory below.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    input_dir = Path("./input_gpu0")
    output_dir = Path("./output_run1_gpu0")

    # Get all .mp4 files
    videos = sorted(input_dir.glob("*.mp4"))

    if not videos:
        logger.error(f"No .mp4 files found in {input_dir}")
        return

    # Process batch
    processor = BatchProcessor(model="resnet50")

    try:
        results = processor.process_batch(
            videos=videos,
            output_dir=output_dir,
            num_gpus=2
        )

        # Return exit code based on success
        failed = sum(1 for r in results if r.get("status") == "failed")
        exit(0 if failed == 0 else 1)

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
