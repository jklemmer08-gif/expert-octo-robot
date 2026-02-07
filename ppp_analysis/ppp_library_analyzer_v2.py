#!/usr/bin/env python3
"""
PPP Library Analyzer v2
Fixed for Stash schema v75+ (uses 'files' array instead of 'file' object)

Outputs:
- tier1_topaz_candidates.csv (highest quality potential - manual review)
- tier2_runpod_vr.csv (VR content for cloud processing)
- tier3_local_bulk.csv (everything else for local processing)
- library_summary.json (overall statistics)

Requirements:
- pip install requests --break-system-packages
"""

import requests
import json
import csv
import os
from datetime import datetime
from collections import defaultdict

# Configuration - UPDATE THESE
STASH_URL = "http://localhost:9999/graphql"
STASH_API_KEY = ""  # Leave empty if no API key required
OUTPUT_DIR = "./ppp_analysis"

# VR Detection patterns
VR_FILENAME_PATTERNS = [
    "_180", "_360", "_vr", "_VR", "_sbs", "_SBS", 
    "_LR", "_TB", "_3dh", "_3DH", "_MONO360",
    "_6k", "_6K", "_7k", "_7K", "_8k", "_8K",
    "_fisheye", "_FISHEYE", "MKX", "POVR", "VRBangers",
    "WankzVR", "VRHush", "SLR", "VirtualReal", "Czech VR",
    "RealJamVR", "VRConk", "VRLatina", "NaughtyAmerica"
]

# Quality thresholds
RESOLUTION_TIERS = {
    "8K+": 7680,
    "6K": 5760,
    "5K": 4800,
    "4K": 3840,
    "1440p": 2560,
    "1080p": 1920,
    "720p": 1280,
    "SD": 0
}

def query_stash(query: str, variables: dict = None) -> dict:
    """Execute GraphQL query against Stash"""
    headers = {"Content-Type": "application/json"}
    if STASH_API_KEY:
        headers["ApiKey"] = STASH_API_KEY
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    try:
        response = requests.post(STASH_URL, json=payload, headers=headers, timeout=60)
        result = response.json()
        if response.status_code != 200:
            print(f"GraphQL Error: {result.get('errors', 'Unknown error')}")
            return None
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error querying Stash: {e}")
        return None

def get_vr_tag_ids() -> list:
    """Find VR-related tag IDs"""
    query = '{ findTags(filter: { per_page: 500 }) { tags { id name } } }'
    result = query_stash(query)
    
    if not result or "data" not in result:
        return []
    
    tags = result["data"]["findTags"]["tags"]
    vr_tag_ids = []
    
    for tag in tags:
        name_lower = tag["name"].lower()
        if "vr" in name_lower or "virtual reality" in name_lower:
            vr_tag_ids.append(tag["id"])
            print(f"  Found VR tag: {tag['name']} (ID: {tag['id']})")
    
    return vr_tag_ids

def get_all_scenes(page_size: int = 100) -> list:
    """Fetch all scenes from Stash with pagination - Schema v75+"""
    all_scenes = []
    page = 1
    
    # Updated query for schema v75+ (files as array)
    query = """
    query FindScenes($filter: FindFilterType) {
        findScenes(filter: $filter) {
            count
            scenes {
                id
                title
                details
                rating100
                o_counter
                play_count
                play_duration
                date
                created_at
                files {
                    path
                    size
                    duration
                    video_codec
                    audio_codec
                    width
                    height
                    frame_rate
                    bit_rate
                }
                studio {
                    name
                }
                performers {
                    name
                }
                tags {
                    id
                    name
                }
            }
        }
    }
    """
    
    while True:
        variables = {
            "filter": {
                "page": page,
                "per_page": page_size,
                "sort": "created_at",
                "direction": "DESC"
            }
        }
        
        result = query_stash(query, variables)
        if not result or "data" not in result:
            print(f"Error fetching page {page}")
            break
            
        scenes_data = result["data"]["findScenes"]
        scenes = scenes_data["scenes"]
        total = scenes_data["count"]
        
        if not scenes:
            break
            
        all_scenes.extend(scenes)
        print(f"  Fetched {len(all_scenes)}/{total} scenes...")
        
        if len(all_scenes) >= total:
            break
            
        page += 1
    
    return all_scenes

def get_primary_file(scene: dict) -> dict:
    """Get the primary (first) file from a scene's files array"""
    files = scene.get("files", [])
    if files and len(files) > 0:
        return files[0]
    return {}

def is_vr_content(scene: dict, vr_tag_ids: list) -> bool:
    """Determine if scene is VR content"""
    # Check tags
    for tag in scene.get("tags", []):
        if tag["id"] in vr_tag_ids or "vr" in tag["name"].lower():
            return True
    
    # Check filename patterns
    file_info = get_primary_file(scene)
    if file_info and file_info.get("path"):
        path_lower = file_info["path"].lower()
        for pattern in VR_FILENAME_PATTERNS:
            if pattern.lower() in path_lower:
                return True
    
    # Check aspect ratio (VR is typically very wide - SBS is 2:1 or wider)
    if file_info:
        width = file_info.get("width", 0)
        height = file_info.get("height", 0)
        if height > 0 and width / height >= 1.9:  # Wider than 16:9
            return True
    
    return False

def get_resolution_tier(width: int) -> str:
    """Categorize resolution into tiers"""
    for tier, min_width in RESOLUTION_TIERS.items():
        if width >= min_width:
            return tier
    return "Unknown"

def calculate_quality_score(scene: dict) -> float:
    """
    Calculate a quality score (0-100) based on multiple factors.
    Higher score = better candidate for premium processing.
    """
    score = 50.0  # Base score
    
    file_info = get_primary_file(scene)
    
    # Resolution factor (0-25 points)
    width = file_info.get("width", 0)
    if width >= 7680:
        score += 25  # Already 8K
    elif width >= 5760:
        score += 20  # 6K
    elif width >= 3840:
        score += 15  # 4K - good upscale candidate
    elif width >= 1920:
        score += 10  # 1080p - decent upscale candidate
    elif width >= 1280:
        score += 5   # 720p - needs work
    
    # Bitrate factor (0-15 points) - higher bitrate = better source
    bitrate = file_info.get("bit_rate", 0)
    if bitrate:
        bitrate_mbps = bitrate / 1_000_000
        if bitrate_mbps >= 50:
            score += 15
        elif bitrate_mbps >= 30:
            score += 10
        elif bitrate_mbps >= 15:
            score += 5
    
    # User engagement factor (0-10 points)
    play_count = scene.get("play_count") or 0
    o_counter = scene.get("o_counter") or 0
    rating = scene.get("rating100") or 0
    
    if o_counter >= 3:
        score += 10
    elif o_counter >= 1:
        score += 7
    elif play_count >= 5:
        score += 5
    elif play_count >= 2:
        score += 3
    
    # Rating bonus (0-5 points)
    if rating:
        score += (rating / 100) * 5
    
    return min(100, max(0, score))

def calculate_upscale_potential(scene: dict, is_vr: bool) -> dict:
    """
    Determine upscaling potential and recommended target.
    """
    file_info = get_primary_file(scene)
    width = file_info.get("width", 0)
    height = file_info.get("height", 0)
    
    result = {
        "current_resolution": f"{width}x{height}",
        "current_tier": get_resolution_tier(width),
        "is_vr": is_vr,
        "can_upscale": False,
        "recommended_target": None,
        "upscale_factor": None,
        "processing_tier": "skip",
        "reason": ""
    }
    
    if width == 0:
        result["reason"] = "No resolution data"
        return result
    
    if is_vr:
        # VR upscaling logic
        if width >= 7680:
            result["reason"] = "Already 8K - no upscale needed"
            result["processing_tier"] = "skip"
        elif width >= 5760:
            result["can_upscale"] = True
            result["recommended_target"] = "8K"
            result["upscale_factor"] = 1.33
            result["processing_tier"] = "tier2_runpod"  # 6K→8K needs cloud
            result["reason"] = "6K VR - upscale to 8K on RunPod"
        elif width >= 3840:
            result["can_upscale"] = True
            result["recommended_target"] = "6K"
            result["upscale_factor"] = 1.5
            result["processing_tier"] = "tier3_local"  # 4K→6K can be local
            result["reason"] = "4K VR - upscale to 6K locally"
        elif width >= 2560:
            result["can_upscale"] = True
            result["recommended_target"] = "5K"
            result["upscale_factor"] = 2.0
            result["processing_tier"] = "tier3_local"
            result["reason"] = "1440p VR - upscale to 5K locally"
        else:
            result["can_upscale"] = True
            result["recommended_target"] = "4K"
            result["upscale_factor"] = 2.0
            result["processing_tier"] = "tier3_local"
            result["reason"] = "Low-res VR - upscale to 4K locally"
    else:
        # 2D content logic
        if width >= 3840:
            result["reason"] = "Already 4K - no upscale needed for 2D"
            result["processing_tier"] = "skip"
        elif width >= 1920:
            result["can_upscale"] = True
            result["recommended_target"] = "4K"
            result["upscale_factor"] = 2.0
            result["processing_tier"] = "tier3_local"
            result["reason"] = "1080p 2D - upscale to 4K locally"
        elif width >= 1280:
            # 720p→1080p is not worth AI upscaling
            result["can_upscale"] = False
            result["recommended_target"] = "1080p"
            result["upscale_factor"] = 1.5
            result["processing_tier"] = "skip"
            result["reason"] = "720p 2D - use FFmpeg lanczos (AI overkill)"
        else:
            result["can_upscale"] = True
            result["recommended_target"] = "1080p"
            result["upscale_factor"] = 2.0
            result["processing_tier"] = "tier3_local"
            result["reason"] = "SD 2D - upscale to 1080p locally"
    
    return result

def analyze_library(scenes: list, vr_tag_ids: list) -> dict:
    """Analyze all scenes and categorize them"""
    analysis = {
        "total_scenes": len(scenes),
        "vr_count": 0,
        "2d_count": 0,
        "by_resolution": defaultdict(int),
        "by_processing_tier": defaultdict(list),
        "top_candidates": [],  # For Tier 1 (Topaz)
        "vr_upscale_candidates": [],  # For Tier 2 (RunPod)
        "local_candidates": [],  # For Tier 3
        "skip_list": [],
        "missing_metadata": []
    }
    
    for i, scene in enumerate(scenes):
        if (i + 1) % 100 == 0:
            print(f"  Analyzing scene {i + 1}/{len(scenes)}...")
        
        file_info = get_primary_file(scene)
        
        # Skip scenes without file info
        if not file_info or not file_info.get("path"):
            analysis["missing_metadata"].append({
                "id": scene["id"],
                "title": scene.get("title", "Unknown")
            })
            continue
        
        # Analyze the scene
        is_vr = is_vr_content(scene, vr_tag_ids)
        quality_score = calculate_quality_score(scene)
        upscale_info = calculate_upscale_potential(scene, is_vr)
        
        # Count VR vs 2D
        if is_vr:
            analysis["vr_count"] += 1
        else:
            analysis["2d_count"] += 1
        
        # Count by resolution
        resolution_tier = get_resolution_tier(file_info.get("width", 0))
        analysis["by_resolution"][resolution_tier] += 1
        
        # Build scene record
        scene_record = {
            "id": scene["id"],
            "title": scene.get("title", "")[:80],  # Truncate long titles
            "path": file_info.get("path", ""),
            "studio": scene.get("studio", {}).get("name", "") if scene.get("studio") else "",
            "performers": ", ".join([p["name"] for p in scene.get("performers", [])])[:60],
            "resolution": f"{file_info.get('width', 0)}x{file_info.get('height', 0)}",
            "resolution_tier": resolution_tier,
            "bitrate_mbps": round(file_info.get("bit_rate", 0) / 1_000_000, 1) if file_info.get("bit_rate") else 0,
            "duration_min": round(file_info.get("duration", 0) / 60, 1) if file_info.get("duration") else 0,
            "size_gb": round(file_info.get("size", 0) / (1024**3), 2) if file_info.get("size") else 0,
            "is_vr": is_vr,
            "rating": scene.get("rating100") or 0,
            "play_count": scene.get("play_count") or 0,
            "o_counter": scene.get("o_counter") or 0,
            "quality_score": round(quality_score, 1),
            "can_upscale": upscale_info["can_upscale"],
            "target_resolution": upscale_info["recommended_target"] or "",
            "processing_tier": upscale_info["processing_tier"],
            "recommendation": upscale_info["reason"]
        }
        
        # Categorize into tiers
        tier = upscale_info["processing_tier"]
        analysis["by_processing_tier"][tier].append(scene_record)
        
        # Special handling for high-quality candidates (Tier 1 - Topaz)
        # VR content with high engagement that would benefit from premium processing
        if is_vr and quality_score >= 65 and upscale_info["can_upscale"]:
            analysis["top_candidates"].append(scene_record)
        
        # VR content for RunPod (Tier 2) - 6K→8K
        if tier == "tier2_runpod":
            analysis["vr_upscale_candidates"].append(scene_record)
        
        # Local processing candidates (Tier 3)
        if tier == "tier3_local":
            analysis["local_candidates"].append(scene_record)
        
        # Skip list
        if tier == "skip":
            analysis["skip_list"].append(scene_record)
    
    # Sort candidates by quality score
    analysis["top_candidates"].sort(key=lambda x: x["quality_score"], reverse=True)
    analysis["vr_upscale_candidates"].sort(key=lambda x: x["quality_score"], reverse=True)
    analysis["local_candidates"].sort(key=lambda x: x["quality_score"], reverse=True)
    
    return analysis

def export_results(analysis: dict, output_dir: str):
    """Export analysis results to CSV and JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    fieldnames = [
        "id", "title", "path", "studio", "performers", "resolution", 
        "resolution_tier", "bitrate_mbps", "duration_min", "size_gb",
        "is_vr", "rating", "play_count", "o_counter", "quality_score",
        "can_upscale", "target_resolution", "processing_tier", "recommendation"
    ]
    
    # Export Tier 1 candidates (Topaz - manual review)
    tier1_path = os.path.join(output_dir, "tier1_topaz_candidates.csv")
    if analysis["top_candidates"]:
        with open(tier1_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analysis["top_candidates"][:50])  # Top 50
        print(f"✓ Exported {min(50, len(analysis['top_candidates']))} Tier 1 candidates to {tier1_path}")
    else:
        print("  No Tier 1 candidates found")
    
    # Export Tier 2 candidates (RunPod VR)
    tier2_path = os.path.join(output_dir, "tier2_runpod_vr.csv")
    if analysis["vr_upscale_candidates"]:
        with open(tier2_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analysis["vr_upscale_candidates"][:100])  # Top 100
        print(f"✓ Exported {min(100, len(analysis['vr_upscale_candidates']))} Tier 2 candidates to {tier2_path}")
    else:
        print("  No Tier 2 candidates found")
    
    # Export Tier 3 candidates (Local bulk)
    tier3_path = os.path.join(output_dir, "tier3_local_bulk.csv")
    if analysis["local_candidates"]:
        with open(tier3_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analysis["local_candidates"])
        print(f"✓ Exported {len(analysis['local_candidates'])} Tier 3 candidates to {tier3_path}")
    else:
        print("  No Tier 3 candidates found")
    
    # Export skip list
    skip_path = os.path.join(output_dir, "skip_no_upscale_needed.csv")
    if analysis["skip_list"]:
        with open(skip_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analysis["skip_list"])
        print(f"✓ Exported {len(analysis['skip_list'])} skip-list items to {skip_path}")
    
    # Export ALL scenes for reference
    all_path = os.path.join(output_dir, "all_scenes_analyzed.csv")
    all_scenes = (analysis["top_candidates"] + analysis["vr_upscale_candidates"] + 
                  analysis["local_candidates"] + analysis["skip_list"])
    if all_scenes:
        with open(all_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_scenes)
        print(f"✓ Exported all {len(all_scenes)} analyzed scenes to {all_path}")
    
    # Export summary JSON
    summary = {
        "analysis_date": datetime.now().isoformat(),
        "total_scenes": analysis["total_scenes"],
        "vr_count": analysis["vr_count"],
        "2d_count": analysis["2d_count"],
        "by_resolution": dict(analysis["by_resolution"]),
        "tier_counts": {
            "tier1_topaz": len(analysis["top_candidates"]),
            "tier2_runpod": len(analysis["vr_upscale_candidates"]),
            "tier3_local": len(analysis["local_candidates"]),
            "skip": len(analysis["skip_list"])
        },
        "missing_metadata_count": len(analysis["missing_metadata"]),
        "processing_estimates": {
            "tier1_hours_local": len(analysis["top_candidates"][:20]) * 15,
            "tier2_cost_runpod": len(analysis["vr_upscale_candidates"][:100]) * 3.5,
            "tier3_hours_local": len(analysis["local_candidates"]) * 5
        }
    }
    
    summary_path = os.path.join(output_dir, "library_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Exported summary to {summary_path}")
    
    return summary

def print_summary(summary: dict):
    """Print a nice summary to console"""
    print("\n" + "="*60)
    print("PPP LIBRARY ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nTotal Scenes: {summary['total_scenes']}")
    print(f"  - VR Content: {summary['vr_count']}")
    print(f"  - 2D Content: {summary['2d_count']}")
    
    print("\nBy Resolution:")
    for tier in ["8K+", "6K", "5K", "4K", "1440p", "1080p", "720p", "SD", "Unknown"]:
        count = summary["by_resolution"].get(tier, 0)
        if count > 0:
            bar_len = min(count // 20 + 1, 30)
            bar = "█" * bar_len
            print(f"  {tier:8s}: {bar} {count}")
    
    print("\nProcessing Tier Recommendations:")
    tiers = summary["tier_counts"]
    print(f"  Tier 1 (Topaz - Premium):  {tiers['tier1_topaz']:4d} scenes")
    print(f"  Tier 2 (RunPod - VR 8K):   {tiers['tier2_runpod']:4d} scenes")
    print(f"  Tier 3 (Local - Bulk):     {tiers['tier3_local']:4d} scenes")
    print(f"  Skip (Already Good):       {tiers['skip']:4d} scenes")
    
    print("\nEstimated Processing:")
    est = summary["processing_estimates"]
    print(f"  Tier 1: ~{est['tier1_hours_local']} hours on RTX 3060 Ti (Topaz)")
    print(f"  Tier 2: ~${est['tier2_cost_runpod']:.0f} on RunPod (top 100)")
    print(f"  Tier 3: ~{est['tier3_hours_local']} hours on Arc B580")
    
    print("\n" + "="*60)
    print(f"Output files written to: {OUTPUT_DIR}")
    print("="*60 + "\n")

def main():
    print("="*60)
    print("PPP Library Analyzer v2")
    print("For Stash Schema v75+")
    print("="*60)
    print(f"\nConnecting to Stash at {STASH_URL}...")
    
    # Test connection
    test_result = query_stash("{ systemStatus { databaseSchema } }")
    if not test_result or "data" not in test_result:
        print("ERROR: Could not connect to Stash. Check URL and API key.")
        return
    
    schema = test_result["data"]["systemStatus"]["databaseSchema"]
    print(f"✓ Connected! Schema version: {schema}")
    
    # Find VR tags
    print("\nLooking for VR tags...")
    vr_tag_ids = get_vr_tag_ids()
    if not vr_tag_ids:
        print("  No VR tags found - will use filename detection only")
    
    # Fetch all scenes
    print("\nFetching scenes from Stash...")
    scenes = get_all_scenes()
    
    if not scenes:
        print("No scenes found. Check your Stash database.")
        return
    
    print(f"\nAnalyzing {len(scenes)} scenes...")
    analysis = analyze_library(scenes, vr_tag_ids)
    
    print("\nExporting results...")
    summary = export_results(analysis, OUTPUT_DIR)
    
    print_summary(summary)
    
    print("Next steps:")
    print("1. Review tier1_topaz_candidates.csv - pick your top 10-20 favorites")
    print("2. Review tier2_runpod_vr.csv - confirm VR content for cloud processing")
    print("3. tier3_local_bulk.csv is ready for automated local processing")
    print("\nHappy processing! 🎬")

if __name__ == "__main__":
    main()
