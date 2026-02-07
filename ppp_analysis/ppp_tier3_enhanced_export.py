#!/usr/bin/env python3
"""
PPP Tier 3 Enhanced Export
Creates a detailed CSV with all filterable columns for manual review.

Columns include:
- All tags (comma-separated)
- Individual tag flags (VR, POV, Anal, etc.)
- All performers (comma-separated)
- Studio
- Quality metrics

Perfect for filtering in Excel/Google Sheets.
"""

import requests
import json
import csv
import os
from datetime import datetime
from collections import defaultdict

# Configuration
STASH_URL = "http://localhost:9999/graphql"
STASH_API_KEY = ""
OUTPUT_DIR = "./ppp_analysis"

# Common tags to create individual filter columns for
# Add any tags you want as separate filterable columns
FILTER_TAGS = [
    "VR", "POV", "Anal", "Blowjob", "Creampie", "Facial", 
    "Threesome", "Lesbian", "MILF", "Teen", "BBC", "Interracial",
    "Bondage", "BDSM", "Solo", "Masturbation", "Squirt",
    "Big Tits", "Big Ass", "Blonde", "Brunette", "Redhead",
    "Asian", "Latina", "Ebony", "European"
]

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

def get_all_scenes_detailed(page_size: int = 100) -> list:
    """Fetch all scenes with full details"""
    all_scenes = []
    page = 1
    
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
                    gender
                    favorite
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
    """Get the primary file from a scene's files array"""
    files = scene.get("files", [])
    return files[0] if files else {}

def has_tag(scene: dict, tag_name: str) -> bool:
    """Check if scene has a specific tag (case-insensitive)"""
    tag_lower = tag_name.lower()
    for tag in scene.get("tags", []):
        if tag_lower in tag["name"].lower():
            return True
    return False

def is_vr_content(scene: dict) -> bool:
    """Determine if scene is VR content"""
    # Check tags
    for tag in scene.get("tags", []):
        if "vr" in tag["name"].lower() or "virtual reality" in tag["name"].lower():
            return True
    
    # Check filename patterns
    vr_patterns = ["_180", "_360", "_vr", "_VR", "_sbs", "_SBS", "_LR", "_TB", 
                   "_6k", "_6K", "_7k", "_7K", "_8k", "_8K", "MKX", "POVR",
                   "VRBangers", "WankzVR", "SLR", "VirtualReal", "CzechVR"]
    
    file_info = get_primary_file(scene)
    if file_info and file_info.get("path"):
        path_lower = file_info["path"].lower()
        for pattern in vr_patterns:
            if pattern.lower() in path_lower:
                return True
    
    # Check aspect ratio
    if file_info:
        width = file_info.get("width", 0)
        height = file_info.get("height", 0)
        if height > 0 and width / height >= 1.9:
            return True
    
    return False

def get_resolution_tier(width: int) -> str:
    """Categorize resolution"""
    tiers = [("8K+", 7680), ("6K", 5760), ("5K", 4800), ("4K", 3840), 
             ("1440p", 2560), ("1080p", 1920), ("720p", 1280), ("SD", 0)]
    for tier, min_width in tiers:
        if width >= min_width:
            return tier
    return "Unknown"

def calculate_quality_score(scene: dict) -> float:
    """Calculate quality score"""
    score = 50.0
    file_info = get_primary_file(scene)
    
    # Resolution
    width = file_info.get("width", 0)
    if width >= 7680: score += 25
    elif width >= 5760: score += 20
    elif width >= 3840: score += 15
    elif width >= 1920: score += 10
    elif width >= 1280: score += 5
    
    # Bitrate
    bitrate = file_info.get("bit_rate", 0)
    if bitrate:
        mbps = bitrate / 1_000_000
        if mbps >= 50: score += 15
        elif mbps >= 30: score += 10
        elif mbps >= 15: score += 5
    
    # Engagement
    o_counter = scene.get("o_counter") or 0
    play_count = scene.get("play_count") or 0
    rating = scene.get("rating100") or 0
    
    if o_counter >= 3: score += 10
    elif o_counter >= 1: score += 7
    elif play_count >= 5: score += 5
    elif play_count >= 2: score += 3
    
    if rating: score += (rating / 100) * 5
    
    return min(100, max(0, score))

def get_upscale_recommendation(scene: dict, is_vr: bool) -> tuple:
    """Get processing tier and recommendation"""
    file_info = get_primary_file(scene)
    width = file_info.get("width", 0)
    
    if width == 0:
        return "skip", "No resolution data"
    
    if is_vr:
        if width >= 7680:
            return "skip", "Already 8K"
        elif width >= 5760:
            return "tier2_runpod", "6K→8K (RunPod)"
        elif width >= 3840:
            return "tier3_local", "4K→6K (Local)"
        elif width >= 2560:
            return "tier3_local", "1440p→5K (Local)"
        else:
            return "tier3_local", "Low-res→4K (Local)"
    else:
        if width >= 3840:
            return "skip", "Already 4K"
        elif width >= 1920:
            return "tier3_local", "1080p→4K (Local)"
        elif width >= 1280:
            return "skip", "720p - use FFmpeg"
        else:
            return "tier3_local", "SD→1080p (Local)"

def main():
    print("="*60)
    print("PPP Tier 3 Enhanced Export")
    print("="*60)
    
    # Test connection
    test_result = query_stash("{ systemStatus { databaseSchema } }")
    if not test_result:
        print("ERROR: Could not connect to Stash")
        return
    print(f"✓ Connected to Stash\n")
    
    # Get all unique tags for reference
    print("Fetching tag list...")
    tag_result = query_stash("{ findTags(filter: { per_page: 1000 }) { tags { name } } }")
    if tag_result and "data" in tag_result:
        all_tags = [t["name"] for t in tag_result["data"]["findTags"]["tags"]]
        print(f"  Found {len(all_tags)} tags in your library")
    
    # Fetch all scenes
    print("\nFetching all scenes...")
    scenes = get_all_scenes_detailed()
    print(f"  Total scenes: {len(scenes)}")
    
    # Build the enhanced records
    print("\nProcessing scenes...")
    records = []
    
    # Track unique values for summary
    all_performers = set()
    all_studios = set()
    tag_counts = defaultdict(int)
    
    for i, scene in enumerate(scenes):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i + 1}/{len(scenes)}...")
        
        file_info = get_primary_file(scene)
        if not file_info:
            continue
        
        is_vr = is_vr_content(scene)
        quality_score = calculate_quality_score(scene)
        tier, recommendation = get_upscale_recommendation(scene, is_vr)
        
        # Get all tag names
        tags = [t["name"] for t in scene.get("tags", [])]
        for tag in tags:
            tag_counts[tag] += 1
        
        # Get performers
        performers = [p["name"] for p in scene.get("performers", [])]
        all_performers.update(performers)
        
        # Get studio
        studio = scene.get("studio", {}).get("name", "") if scene.get("studio") else ""
        if studio:
            all_studios.add(studio)
        
        # Build record
        record = {
            "id": scene["id"],
            "title": scene.get("title", "")[:100],
            "path": file_info.get("path", ""),
            
            # Filtering columns
            "studio": studio,
            "performers": " | ".join(performers),  # Pipe-separated for easier filtering
            "performer_count": len(performers),
            "all_tags": " | ".join(tags),  # Pipe-separated
            "tag_count": len(tags),
            
            # Content type
            "content_type": "VR" if is_vr else "Flat",
            "is_vr": is_vr,
            
            # Resolution info
            "width": file_info.get("width", 0),
            "height": file_info.get("height", 0),
            "resolution": f"{file_info.get('width', 0)}x{file_info.get('height', 0)}",
            "resolution_tier": get_resolution_tier(file_info.get("width", 0)),
            
            # Quality metrics
            "bitrate_mbps": round(file_info.get("bit_rate", 0) / 1_000_000, 1) if file_info.get("bit_rate") else 0,
            "duration_min": round(file_info.get("duration", 0) / 60, 1) if file_info.get("duration") else 0,
            "size_gb": round(file_info.get("size", 0) / (1024**3), 2) if file_info.get("size") else 0,
            "codec": file_info.get("video_codec", ""),
            
            # Engagement
            "rating": scene.get("rating100") or 0,
            "play_count": scene.get("play_count") or 0,
            "o_counter": scene.get("o_counter") or 0,
            "quality_score": round(quality_score, 1),
            
            # Processing
            "processing_tier": tier,
            "recommendation": recommendation,
        }
        
        # Add individual tag columns for common filters
        for filter_tag in FILTER_TAGS:
            col_name = f"tag_{filter_tag.lower().replace(' ', '_')}"
            record[col_name] = "Yes" if has_tag(scene, filter_tag) else ""
        
        records.append(record)
    
    # Sort by quality score
    records.sort(key=lambda x: x["quality_score"], reverse=True)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define fieldnames
    base_fields = [
        "id", "title", "studio", "performers", "performer_count",
        "all_tags", "tag_count", "content_type", "is_vr",
        "width", "height", "resolution", "resolution_tier",
        "bitrate_mbps", "duration_min", "size_gb", "codec",
        "rating", "play_count", "o_counter", "quality_score",
        "processing_tier", "recommendation", "path"
    ]
    tag_fields = [f"tag_{t.lower().replace(' ', '_')}" for t in FILTER_TAGS]
    fieldnames = base_fields + tag_fields
    
    # Export FULL library (all scenes)
    full_path = os.path.join(OUTPUT_DIR, "full_library_filterable.csv")
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n✓ Exported FULL library ({len(records)} scenes) to {full_path}")
    
    # Export Tier 3 only
    tier3_records = [r for r in records if r["processing_tier"] == "tier3_local"]
    tier3_path = os.path.join(OUTPUT_DIR, "tier3_filterable.csv")
    with open(tier3_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tier3_records)
    print(f"✓ Exported Tier 3 ({len(tier3_records)} scenes) to {tier3_path}")
    
    # Export VR-only subset
    vr_records = [r for r in records if r["is_vr"]]
    vr_path = os.path.join(OUTPUT_DIR, "vr_only_filterable.csv")
    with open(vr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(vr_records)
    print(f"✓ Exported VR only ({len(vr_records)} scenes) to {vr_path}")
    
    # Export POV-tagged subset (VR + POV)
    pov_vr_records = [r for r in records if r["is_vr"] or has_tag({"tags": [{"name": t} for t in r["all_tags"].split(" | ") if t]}, "POV")]
    pov_path = os.path.join(OUTPUT_DIR, "vr_and_pov_filterable.csv")
    with open(pov_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([r for r in records if r["is_vr"] or r.get("tag_pov") == "Yes"])
    print(f"✓ Exported VR + POV tagged scenes to {pov_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    
    print(f"\nTotal scenes exported: {len(records)}")
    print(f"  - VR scenes: {len(vr_records)}")
    print(f"  - Flat scenes: {len(records) - len(vr_records)}")
    print(f"  - Tier 3 (processable): {len(tier3_records)}")
    
    print(f"\nUnique performers: {len(all_performers)}")
    print(f"Unique studios: {len(all_studios)}")
    
    print("\nTop 15 tags by frequency:")
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    for tag, count in sorted_tags:
        print(f"  {tag}: {count}")
    
    print("\n" + "="*60)
    print("FILES CREATED:")
    print("="*60)
    print(f"  1. {full_path}")
    print(f"     → Full library with all filter columns")
    print(f"  2. {tier3_path}")
    print(f"     → Just Tier 3 scenes (need processing)")
    print(f"  3. {vr_path}")
    print(f"     → VR content only")
    print(f"  4. {pov_path}")
    print(f"     → VR + POV tagged scenes")
    
    print("\n" + "="*60)
    print("HOW TO FILTER IN EXCEL/SHEETS:")
    print("="*60)
    print("""
1. Open the CSV in Excel or Google Sheets
2. Select all data → Data → Create Filter
3. Filter examples:
   - content_type = "VR" (VR only)
   - tag_pov = "Yes" (POV scenes)
   - studio contains "Brazzers" (specific studio)
   - performers contains "Angela White" (specific performer)
   - quality_score > 70 (high engagement)
   - resolution_tier = "4K" (specific resolution)
   
4. Combine filters:
   - content_type = "VR" AND quality_score > 65
   - tag_pov = "Yes" AND resolution_tier = "1080p"
   
5. Sort by quality_score DESC to see best candidates first
""")

if __name__ == "__main__":
    main()
