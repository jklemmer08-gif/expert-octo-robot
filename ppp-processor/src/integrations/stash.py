"""Stash GraphQL integration — refactored from integrations.py:StashIntegration.

Provides the same API methods for scene lookup, tagging, and scanning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("ppp.integrations.stash")


class StashClient:
    """Stash GraphQL API client for metadata management."""

    def __init__(self, url: str = "http://localhost:9999/graphql",
                 api_key: str = ""):
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["ApiKey"] = api_key

    def _query(self, query: str, variables: dict = None) -> dict:
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        response = requests.post(
            self.url, json=payload, headers=self.headers, timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Stash query failed: {response.text}")
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL error: {data['errors']}")
        return data.get("data", {})

    def find_scene_by_path(self, file_path: str) -> Optional[Dict]:
        result = self._query("""
            query($path: String!) {
                findScenes(scene_filter: {path: {value: $path, modifier: EQUALS}}) {
                    scenes { id title details studio { id name }
                             performers { id name } tags { id name } files { path } }
                }
            }
        """, {"path": file_path})
        scenes = result.get("findScenes", {}).get("scenes", [])
        return scenes[0] if scenes else None

    def find_scene_by_id(self, scene_id: str) -> Optional[Dict]:
        result = self._query("""
            query($id: ID!) {
                findScene(id: $id) {
                    id title details studio { id name }
                    performers { id name } tags { id name } files { path }
                }
            }
        """, {"id": scene_id})
        return result.get("findScene")

    def get_or_create_tag(self, tag_name: str) -> str:
        result = self._query("""
            query($name: String!) {
                findTags(tag_filter: {name: {value: $name, modifier: EQUALS}}) {
                    tags { id name }
                }
            }
        """, {"name": tag_name})
        tags = result.get("findTags", {}).get("tags", [])
        if tags:
            return tags[0]["id"]
        result = self._query("""
            mutation($input: TagCreateInput!) {
                tagCreate(input: $input) { id }
            }
        """, {"input": {"name": tag_name}})
        return result.get("tagCreate", {}).get("id")

    def add_tag_to_scene(self, scene_id: str, tag_name: str) -> bool:
        scene = self.find_scene_by_id(scene_id)
        if not scene:
            return False
        current_ids = [t["id"] for t in scene.get("tags", [])]
        new_id = self.get_or_create_tag(tag_name)
        if new_id in current_ids:
            return True
        current_ids.append(new_id)
        self._query("""
            mutation($input: SceneUpdateInput!) {
                sceneUpdate(input: $input) { id }
            }
        """, {"input": {"id": scene_id, "tag_ids": current_ids}})
        return True

    def find_scenes_by_tag(self, tag_name: str) -> List[Dict]:
        """Find all scenes that have a specific tag."""
        tag_id = self.get_or_create_tag(tag_name)
        if not tag_id:
            return []
        result = self._query("""
            query($tag_id: [ID!]!) {
                findScenes(scene_filter: {tags: {value: $tag_id, modifier: INCLUDES}},
                           filter: {per_page: 100}) {
                    scenes { id title tags { id name } files { path } }
                }
            }
        """, {"tag_id": [tag_id]})
        return result.get("findScenes", {}).get("scenes", [])

    def remove_tag_from_scene(self, scene_id: str, tag_name: str) -> bool:
        """Remove a specific tag from a scene."""
        scene = self.find_scene_by_id(scene_id)
        if not scene:
            return False
        tag_id = self.get_or_create_tag(tag_name)
        current_ids = [t["id"] for t in scene.get("tags", []) if t["id"] != tag_id]
        self._query("""
            mutation($input: SceneUpdateInput!) {
                sceneUpdate(input: $input) { id }
            }
        """, {"input": {"id": scene_id, "tag_ids": current_ids}})
        return True

    def update_scene_after_upscale(self, scene_id: str, new_path: Path,
                                    resolution: str = ""):
        self.add_tag_to_scene(scene_id, "Upscaled")
        self.add_tag_to_scene(scene_id, "PPP-Processed")
        if resolution:
            self.add_tag_to_scene(scene_id, resolution)
        scene = self.find_scene_by_id(scene_id)
        if scene:
            details = scene.get("details", "") or ""
            if "Upscaled by PPP" not in details:
                new_details = (
                    f"{details}\n\nUpscaled by PPP to {resolution}".strip()
                    if details else f"Upscaled by PPP to {resolution}"
                )
                self._query("""
                    mutation($input: SceneUpdateInput!) {
                        sceneUpdate(input: $input) { id }
                    }
                """, {"input": {"id": scene_id, "details": new_details}})

    def trigger_scan(self, paths: Optional[List[str]] = None) -> bool:
        try:
            input_data = {}
            if paths:
                input_data["paths"] = paths
            self._query("""
                mutation($input: ScanMetadataInput!) { metadataScan(input: $input) }
            """, {"input": input_data})
            return True
        except Exception as e:
            logger.warning("Stash scan failed: %s", e)
            return False
