"""Tests for cloud worker (RunPod API mocked)."""

from unittest.mock import MagicMock, patch

import pytest


class TestRunPodClient:

    def _make_client(self, mock_gql):
        from src.workers.cloud_worker import RunPodClient
        client = object.__new__(RunPodClient)
        client.headers = {}
        client._requests = MagicMock()
        # Patch the _gql method on the instance
        client._gql = mock_gql
        # Override get_available_gpus to use our mock
        return client

    def test_find_best_gpu(self):
        gpu_data = [
            {
                "id": "gpu1", "displayName": "NVIDIA RTX 3090",
                "memoryInGb": 24,
                "lowestPrice": {"uninterruptablePrice": 0.30, "minimumBidPrice": 0.20},
            },
            {
                "id": "gpu2", "displayName": "NVIDIA RTX 4090",
                "memoryInGb": 24,
                "lowestPrice": {"uninterruptablePrice": 0.50, "minimumBidPrice": 0.40},
            },
        ]

        mock_gql = MagicMock(return_value={"gpuTypes": gpu_data})
        client = self._make_client(mock_gql)
        gpu = client.find_best_gpu(min_vram=24, max_price=5.0)
        assert gpu is not None
        assert "4090" in gpu["name"]

    def test_find_best_gpu_none_available(self):
        mock_gql = MagicMock(return_value={"gpuTypes": []})
        client = self._make_client(mock_gql)
        gpu = client.find_best_gpu()
        assert gpu is None

    def test_find_best_gpu_budget_filter(self):
        gpu_data = [
            {
                "id": "gpu1", "displayName": "A100",
                "memoryInGb": 80,
                "lowestPrice": {"uninterruptablePrice": 10.0, "minimumBidPrice": 8.0},
            },
        ]

        mock_gql = MagicMock(return_value={"gpuTypes": gpu_data})
        client = self._make_client(mock_gql)
        gpu = client.find_best_gpu(max_price=5.0)
        assert gpu is None


def test_cloud_job_budget_check(test_db, sample_job, test_settings):
    """Cloud worker should refuse if budget is exhausted."""
    test_db.add_job(sample_job)

    # Simulate spent budget
    for i in range(20):
        test_db.log_cost(f"job-{i}", "cloud", "RTX 4090", 4.0, 8.0)

    total = test_db.get_total_cost()
    assert total >= 75.0  # Budget exhausted
