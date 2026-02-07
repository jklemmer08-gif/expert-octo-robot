"""Jellyfin integration — refactored from integrations.py:JellyfinIntegration.

Handles library organization and scanning.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("ppp.integrations.jellyfin")


class JellyfinClient:
    """Jellyfin API client for library management."""

    def __init__(self, url: str = "http://localhost:8096",
                 api_key: str = "", library_path: str = ""):
        self.url = url
        self.headers = {}
        if api_key:
            self.headers["X-Emby-Token"] = api_key
        self.library_path = Path(library_path) if library_path else None

    def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        url = f"{self.url}{endpoint}"
        if method == "GET":
            resp = requests.get(url, headers=self.headers, params=data, timeout=30)
        elif method == "POST":
            resp = requests.post(url, headers=self.headers, json=data, timeout=30)
        else:
            raise ValueError(f"Unknown method: {method}")
        if resp.status_code not in (200, 204):
            raise RuntimeError(f"Jellyfin request failed: {resp.text}")
        return resp.json() if resp.text else {}

    def organize_for_jellyfin(self, source_path: Path,
                               studio: str = "", is_vr: bool = False) -> Path:
        """Determine the target path for Jellyfin library organization."""
        if not self.library_path:
            return source_path

        if is_vr:
            target_dir = self.library_path / "VR"
        elif studio:
            safe = "".join(c for c in studio if c.isalnum() or c in " -_").strip()
            target_dir = self.library_path / "Studios" / safe
        else:
            target_dir = self.library_path / "Unsorted"

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name

        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
            counter += 1

        return target_path

    def organize_and_copy(self, source_path: Path, is_vr: bool = False,
                           studio: str = "", move: bool = False) -> Optional[Path]:
        """Organize and copy/move file to Jellyfin library."""
        target = self.organize_for_jellyfin(source_path, studio=studio, is_vr=is_vr)
        if target == source_path:
            return None

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(source_path), str(target))
            else:
                shutil.copy2(str(source_path), str(target))
            logger.info("Copied to Jellyfin: %s", target)
            return target
        except Exception as e:
            logger.error("Failed to copy to Jellyfin: %s", e)
            return None

    def trigger_library_scan(self) -> bool:
        try:
            self._request("POST", "/Library/Refresh")
            logger.info("Jellyfin library scan triggered")
            return True
        except Exception as e:
            logger.warning("Jellyfin scan failed: %s", e)
            return False

    def search_item(self, query: str) -> List[Dict]:
        try:
            result = self._request("GET", "/Items", {
                "searchTerm": query, "Recursive": True,
                "IncludeItemTypes": "Movie,Video",
            })
            return result.get("Items", [])
        except Exception:
            return []
