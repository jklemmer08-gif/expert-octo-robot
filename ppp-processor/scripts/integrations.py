#!/usr/bin/env python3
"""
PPP Integrations - Stash and Jellyfin integration
Handles post-processing updates and library organization
"""

import requests
import json
import shutil
import os
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import argparse

@dataclass
class StashConfig:
    url: str = "http://localhost:9999/graphql"
    api_key: str = ""

@dataclass  
class JellyfinConfig:
    url: str = "http://localhost:8096"
    api_key: str = ""
    library_path: str = ""  # e.g., /media/jellyfin/VR
    user_id: str = ""

class StashIntegration:
    """Stash GraphQL integration for metadata management"""
    
    def __init__(self, config: Optional[StashConfig] = None):
        self.config = config or StashConfig()
        self.headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            self.headers["ApiKey"] = self.config.api_key
    
    def _query(self, query: str, variables: dict = None) -> dict:
        """Execute GraphQL query"""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        response = requests.post(
            self.config.url, 
            json=payload, 
            headers=self.headers,
            timeout=30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Stash query failed: {response.text}")
        
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL error: {data['errors']}")
        
        return data.get("data", {})
    
    def find_scene_by_path(self, file_path: str) -> Optional[Dict]:
        """Find scene by file path"""
        query = """
        query FindSceneByPath($path: String!) {
            findScenes(scene_filter: {path: {value: $path, modifier: EQUALS}}) {
                scenes {
                    id
                    title
                    details
                    rating100
                    o_counter
                    play_count
                    studio { id name }
                    performers { id name }
                    tags { id name }
                    files { path }
                }
            }
        }
        """
        
        result = self._query(query, {"path": file_path})
        scenes = result.get("findScenes", {}).get("scenes", [])
        return scenes[0] if scenes else None
    
    def find_scene_by_id(self, scene_id: str) -> Optional[Dict]:
        """Find scene by ID"""
        query = """
        query FindScene($id: ID!) {
            findScene(id: $id) {
                id
                title
                details
                rating100
                o_counter
                play_count
                studio { id name }
                performers { id name }
                tags { id name }
                files { path }
            }
        }
        """
        
        result = self._query(query, {"id": scene_id})
        return result.get("findScene")
    
    def get_or_create_tag(self, tag_name: str) -> str:
        """Get tag ID, create if doesn't exist"""
        # First try to find
        query = """
        query FindTag($name: String!) {
            findTags(tag_filter: {name: {value: $name, modifier: EQUALS}}) {
                tags { id name }
            }
        }
        """
        
        result = self._query(query, {"name": tag_name})
        tags = result.get("findTags", {}).get("tags", [])
        
        if tags:
            return tags[0]["id"]
        
        # Create tag
        mutation = """
        mutation CreateTag($input: TagCreateInput!) {
            tagCreate(input: $input) { id name }
        }
        """
        
        result = self._query(mutation, {"input": {"name": tag_name}})
        return result.get("tagCreate", {}).get("id")
    
    def update_scene_file(self, scene_id: str, new_path: str) -> bool:
        """
        Update scene to point to new file path.
        Note: This requires Stash to rescan. We add the file and let Stash merge.
        """
        # Stash doesn't have a direct "update file path" mutation
        # The recommended approach is:
        # 1. Move/copy file to Stash library
        # 2. Run a scan
        # 3. Stash will auto-detect based on filename similarity
        
        # For now, we'll just add an "Upscaled" tag and log the new path
        print(f"Note: Move {new_path} to Stash library and run scan")
        return True
    
    def add_tag_to_scene(self, scene_id: str, tag_name: str) -> bool:
        """Add a tag to a scene"""
        # Get current tags
        scene = self.find_scene_by_id(scene_id)
        if not scene:
            return False
        
        current_tag_ids = [t["id"] for t in scene.get("tags", [])]
        
        # Get or create the new tag
        new_tag_id = self.get_or_create_tag(tag_name)
        
        if new_tag_id in current_tag_ids:
            return True  # Already has tag
        
        current_tag_ids.append(new_tag_id)
        
        # Update scene
        mutation = """
        mutation UpdateScene($input: SceneUpdateInput!) {
            sceneUpdate(input: $input) { id }
        }
        """
        
        self._query(mutation, {
            "input": {
                "id": scene_id,
                "tag_ids": current_tag_ids
            }
        })
        
        return True
    
    def update_scene_after_upscale(self, scene_id: str, new_path: Path, 
                                    resolution: str = "") -> Dict:
        """
        Post-processing update for upscaled scene.
        - Adds "Upscaled" tag
        - Adds resolution tag (e.g., "6K", "8K")
        - Updates details with processing info
        """
        print(f"Updating Stash scene {scene_id}...")
        
        # Add tags
        self.add_tag_to_scene(scene_id, "Upscaled")
        self.add_tag_to_scene(scene_id, "PPP-Processed")
        
        if resolution:
            self.add_tag_to_scene(scene_id, resolution)
        
        # Get current scene details
        scene = self.find_scene_by_id(scene_id)
        if scene:
            details = scene.get("details", "") or ""
            if "Upscaled by PPP" not in details:
                new_details = f"{details}\n\nUpscaled by PPP to {resolution}" if details else f"Upscaled by PPP to {resolution}"
                
                mutation = """
                mutation UpdateScene($input: SceneUpdateInput!) {
                    sceneUpdate(input: $input) { id }
                }
                """
                
                self._query(mutation, {
                    "input": {
                        "id": scene_id,
                        "details": new_details.strip()
                    }
                })
        
        return {"success": True, "scene_id": scene_id, "new_path": str(new_path)}
    
    def trigger_scan(self, paths: List[str] = None) -> bool:
        """Trigger Stash library scan"""
        mutation = """
        mutation MetadataScan($input: ScanMetadataInput!) {
            metadataScan(input: $input)
        }
        """
        
        input_data = {}
        if paths:
            input_data["paths"] = paths
        
        try:
            self._query(mutation, {"input": input_data})
            return True
        except Exception as e:
            print(f"Scan trigger failed: {e}")
            return False


class JellyfinIntegration:
    """Jellyfin integration for library management"""
    
    def __init__(self, config: Optional[JellyfinConfig] = None):
        self.config = config or JellyfinConfig()
        self.headers = {}
        if self.config.api_key:
            self.headers["X-Emby-Token"] = self.config.api_key
    
    def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make API request to Jellyfin"""
        url = f"{self.config.url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, headers=self.headers, params=data)
        elif method == "POST":
            response = requests.post(url, headers=self.headers, json=data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if response.status_code not in (200, 204):
            raise RuntimeError(f"Jellyfin request failed: {response.text}")
        
        return response.json() if response.text else {}
    
    def get_library_path(self, library_name: str = "VR") -> Optional[Path]:
        """Get the filesystem path for a Jellyfin library"""
        # This requires knowing the library configuration
        # For now, use the configured path
        if self.config.library_path:
            return Path(self.config.library_path)
        return None
    
    def organize_for_jellyfin(self, source_path: Path, 
                              studio: str = "", 
                              title: str = "",
                              is_vr: bool = False) -> Path:
        """
        Organize file for Jellyfin library structure.
        
        Structure:
        library_root/
        ├── Studios/
        │   └── StudioName/
        │       └── video.mp4
        └── VR/
            └── video_180_sbs.mp4
        """
        library_root = self.get_library_path()
        
        if not library_root:
            return source_path
        
        # Determine target directory
        if is_vr:
            target_dir = library_root / "VR"
        elif studio:
            # Clean studio name for filesystem
            safe_studio = "".join(c for c in studio if c.isalnum() or c in " -_").strip()
            target_dir = library_root / "Studios" / safe_studio
        else:
            target_dir = library_root / "Unsorted"
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate target filename
        target_name = source_path.name
        if title and title != source_path.stem:
            # Use title if different from filename
            safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:100]
            target_name = f"{safe_title}{source_path.suffix}"
        
        target_path = target_dir / target_name
        
        # Handle duplicates
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{target_path.stem}_{counter}{target_path.suffix}"
            counter += 1
        
        return target_path
    
    def copy_to_library(self, source_path: Path, target_path: Path, 
                        move: bool = False) -> bool:
        """Copy or move file to Jellyfin library"""
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            if move:
                shutil.move(str(source_path), str(target_path))
            else:
                shutil.copy2(str(source_path), str(target_path))
            
            print(f"{'Moved' if move else 'Copied'} to: {target_path}")
            return True
            
        except Exception as e:
            print(f"Error copying to library: {e}")
            return False
    
    def trigger_library_scan(self) -> bool:
        """Trigger Jellyfin library scan"""
        try:
            self._request("POST", "/Library/Refresh")
            print("Jellyfin library scan triggered")
            return True
        except Exception as e:
            print(f"Could not trigger scan: {e}")
            return False
    
    def search_item(self, query: str) -> List[Dict]:
        """Search for items in Jellyfin"""
        try:
            result = self._request("GET", "/Items", {
                "searchTerm": query,
                "Recursive": True,
                "IncludeItemTypes": "Movie,Video"
            })
            return result.get("Items", [])
        except Exception:
            return []


class PPPIntegrations:
    """Combined integration manager"""
    
    def __init__(self, stash_url: str = "http://localhost:9999/graphql",
                 stash_api_key: str = "",
                 jellyfin_url: str = "http://localhost:8096",
                 jellyfin_api_key: str = "",
                 jellyfin_library: str = ""):
        
        self.stash = StashIntegration(StashConfig(
            url=stash_url,
            api_key=stash_api_key
        ))
        
        self.jellyfin = JellyfinIntegration(JellyfinConfig(
            url=jellyfin_url,
            api_key=jellyfin_api_key,
            library_path=jellyfin_library
        ))
    
    def post_process(self, source_path: Path, output_path: Path,
                     scene_id: str = "", resolution: str = "",
                     is_vr: bool = False, studio: str = "",
                     copy_to_jellyfin: bool = True) -> Dict:
        """
        Full post-processing workflow:
        1. Update Stash with tags and metadata
        2. Copy/organize for Jellyfin
        3. Trigger library scans
        """
        result = {
            "source": str(source_path),
            "output": str(output_path),
            "stash_updated": False,
            "jellyfin_copied": False,
            "final_path": str(output_path)
        }
        
        # Update Stash
        if scene_id:
            try:
                self.stash.update_scene_after_upscale(scene_id, output_path, resolution)
                result["stash_updated"] = True
            except Exception as e:
                print(f"Stash update failed: {e}")
        
        # Copy to Jellyfin library
        if copy_to_jellyfin and self.jellyfin.config.library_path:
            target_path = self.jellyfin.organize_for_jellyfin(
                output_path, studio=studio, is_vr=is_vr
            )
            
            if self.jellyfin.copy_to_library(output_path, target_path):
                result["jellyfin_copied"] = True
                result["final_path"] = str(target_path)
        
        return result
    
    def trigger_scans(self) -> Dict:
        """Trigger scans on both Stash and Jellyfin"""
        return {
            "stash_scan": self.stash.trigger_scan(),
            "jellyfin_scan": self.jellyfin.trigger_library_scan()
        }


def main():
    parser = argparse.ArgumentParser(description="PPP Integrations - Stash/Jellyfin")
    parser.add_argument("--stash-url", default="http://localhost:9999/graphql")
    parser.add_argument("--stash-key", default="")
    parser.add_argument("--jellyfin-url", default="http://localhost:8096")
    parser.add_argument("--jellyfin-key", default="")
    parser.add_argument("--jellyfin-library", default="")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Test connection
    test_parser = subparsers.add_parser("test", help="Test connections")
    
    # Find scene
    find_parser = subparsers.add_parser("find", help="Find scene in Stash")
    find_parser.add_argument("path", help="File path to search")
    
    # Update scene
    update_parser = subparsers.add_parser("update", help="Update scene after upscale")
    update_parser.add_argument("--scene-id", required=True)
    update_parser.add_argument("--new-path", required=True)
    update_parser.add_argument("--resolution", default="")
    
    # Scan
    scan_parser = subparsers.add_parser("scan", help="Trigger library scans")
    
    args = parser.parse_args()
    
    integrations = PPPIntegrations(
        stash_url=args.stash_url,
        stash_api_key=args.stash_key,
        jellyfin_url=args.jellyfin_url,
        jellyfin_api_key=args.jellyfin_key,
        jellyfin_library=args.jellyfin_library
    )
    
    if args.command == "test":
        print("Testing Stash connection...")
        try:
            result = integrations.stash._query("{ systemStatus { databaseSchema } }")
            print(f"  ✓ Stash connected (schema: {result['systemStatus']['databaseSchema']})")
        except Exception as e:
            print(f"  ✗ Stash failed: {e}")
        
        print("\nTesting Jellyfin connection...")
        try:
            result = integrations.jellyfin._request("GET", "/System/Info")
            print(f"  ✓ Jellyfin connected ({result.get('ServerName', 'Unknown')})")
        except Exception as e:
            print(f"  ✗ Jellyfin failed: {e}")
    
    elif args.command == "find":
        scene = integrations.stash.find_scene_by_path(args.path)
        if scene:
            print(f"Found scene: {scene['id']}")
            print(f"  Title: {scene.get('title', 'N/A')}")
            print(f"  Studio: {scene.get('studio', {}).get('name', 'N/A')}")
            print(f"  Tags: {', '.join(t['name'] for t in scene.get('tags', []))}")
        else:
            print("Scene not found")
    
    elif args.command == "update":
        result = integrations.stash.update_scene_after_upscale(
            args.scene_id, 
            Path(args.new_path),
            args.resolution
        )
        print(f"Update result: {result}")
    
    elif args.command == "scan":
        result = integrations.trigger_scans()
        print(f"Scan results: {result}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
