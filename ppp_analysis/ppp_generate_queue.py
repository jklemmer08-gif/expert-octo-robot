#!/usr/bin/env python3
"""
PPP Processing Queue Generator
Parses scene IDs and generates a processing queue with file paths.
"""

import requests
import json
import csv
import os

STASH_URL = "http://localhost:9999/graphql"
STASH_API_KEY = ""
OUTPUT_DIR = "./ppp_analysis"

# The concatenated IDs from your spreadsheet
RAW_IDS = """260426122607262826202622261126012515206026742672262326162529251425072367938202418171784898882858825805756462434244422292188217921742167167286085526862687259325882576254025382480245024112404240524082385222822142206220021972168215420281830169116671612152614271410129312561210121212131216121712191221119912051207118311841188118911911194119511811182113210901025100710099799829679689699669429069078998868698688598617947287317037057076856536376306126055615294874754744584674484454233963853803313373263082882842732802722542512552592622302112081881971651701621341201101121188178677045494335282459245216621058105010479919539527821150114211081042998949950787788199139129"""

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
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_all_valid_ids() -> set:
    """Get all valid scene IDs from Stash"""
    query = """
    query { 
        findScenes(filter: { per_page: -1 }) { 
            scenes { id } 
        } 
    }
    """
    result = query_stash(query)
    if result and "data" in result:
        return {s["id"] for s in result["data"]["findScenes"]["scenes"]}
    return set()

def parse_concatenated_ids(raw: str, valid_ids: set) -> list:
    """
    Parse concatenated IDs by trying different split strategies.
    Uses valid_ids from Stash to validate.
    """
    raw = raw.strip().replace(" ", "").replace("\n", "")
    found_ids = []
    
    i = 0
    while i < len(raw):
        # Try 4-digit first, then 3-digit, then 2-digit, then 1-digit
        matched = False
        for length in [4, 3, 2, 1]:
            if i + length <= len(raw):
                candidate = raw[i:i+length]
                if candidate in valid_ids:
                    found_ids.append(candidate)
                    i += length
                    matched = True
                    break
        
        if not matched:
            # Skip this character and try again
            i += 1
    
    return found_ids

def get_scenes_by_ids(ids: list) -> list:
    """Fetch full scene details for given IDs - one at a time"""
    scenes = []
    
    query = """
    query FindScene($id: ID!) {
        findScene(id: $id) {
            id
            title
            files {
                path
                width
                height
                duration
                bit_rate
                size
            }
            studio { name }
            performers { name }
            tags { name }
            rating100
            o_counter
            play_count
        }
    }
    """
    
    for i, scene_id in enumerate(ids):
        result = query_stash(query, {"id": scene_id})
        if result and "data" in result and result["data"]["findScene"]:
            scenes.append(result["data"]["findScene"])
        
        if (i + 1) % 25 == 0:
            print(f"  Fetched {i+1}/{len(ids)} scenes...")
    
    print(f"  Fetched {len(scenes)}/{len(ids)} scenes...")
    return scenes

def main():
    print("="*60)
    print("PPP Processing Queue Generator")
    print("="*60)
    
    # Get all valid IDs from Stash
    print("\nFetching valid scene IDs from Stash...")
    valid_ids = get_all_valid_ids()
    print(f"  Found {len(valid_ids)} scenes in Stash")
    
    # Parse the concatenated IDs
    print("\nParsing your ID list...")
    parsed_ids = parse_concatenated_ids(RAW_IDS, valid_ids)
    print(f"  Parsed {len(parsed_ids)} valid scene IDs")
    
    # Remove duplicates while preserving order
    unique_ids = list(dict.fromkeys(parsed_ids))
    print(f"  Unique IDs: {len(unique_ids)}")
    
    # Fetch scene details
    print("\nFetching scene details...")
    scenes = get_scenes_by_ids(unique_ids)
    print(f"  Retrieved {len(scenes)} scenes")
    
    # Build processing queue
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    queue = []
    vr_count = 0
    flat_count = 0
    
    for scene in scenes:
        files = scene.get("files", [])
        if not files:
            continue
        
        file_info = files[0]
        path = file_info.get("path", "")
        width = file_info.get("width", 0)
        
        # Detect VR
        is_vr = False
        vr_patterns = ["_180", "_360", "_vr", "_sbs", "_LR", "_TB", "_6k", "_8k", "VR"]
        for pattern in vr_patterns:
            if pattern.lower() in path.lower():
                is_vr = True
                break
        
        # Check tags for VR
        for tag in scene.get("tags", []):
            if "vr" in tag["name"].lower():
                is_vr = True
                break
        
        if is_vr:
            vr_count += 1
        else:
            flat_count += 1
        
        # Determine target resolution
        if is_vr:
            if width >= 5760:
                target = "8K"
                tier = "runpod"
            elif width >= 3840:
                target = "6K"
                tier = "local"
            else:
                target = "4K"
                tier = "local"
        else:
            if width >= 1920:
                target = "4K"
                tier = "local"
            else:
                target = "1080p"
                tier = "local"
        
        queue.append({
            "id": scene["id"],
            "title": scene.get("title", "")[:80],
            "path": path,
            "studio": scene.get("studio", {}).get("name", "") if scene.get("studio") else "",
            "performers": " | ".join([p["name"] for p in scene.get("performers", [])]),
            "width": width,
            "height": file_info.get("height", 0),
            "duration_min": round(file_info.get("duration", 0) / 60, 1),
            "size_gb": round(file_info.get("size", 0) / (1024**3), 2),
            "is_vr": is_vr,
            "content_type": "VR" if is_vr else "Flat",
            "target_resolution": target,
            "processing_tier": tier,
            "rating": scene.get("rating100") or 0,
            "o_counter": scene.get("o_counter") or 0,
            "play_count": scene.get("play_count") or 0,
        })
    
    # Sort by VR first, then by o_counter
    queue.sort(key=lambda x: (not x["is_vr"], -(x["o_counter"] or 0)))
    
    # Export CSV
    csv_path = os.path.join(OUTPUT_DIR, "processing_queue.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=queue[0].keys())
        writer.writeheader()
        writer.writerows(queue)
    print(f"\n✓ Exported processing queue to {csv_path}")
    
    # Export JSON (for PPP processor)
    json_path = os.path.join(OUTPUT_DIR, "processing_queue.json")
    with open(json_path, "w") as f:
        json.dump(queue, f, indent=2)
    print(f"✓ Exported JSON queue to {json_path}")
    
    # Export simple file list (one path per line)
    paths_path = os.path.join(OUTPUT_DIR, "file_paths.txt")
    with open(paths_path, "w") as f:
        for item in queue:
            f.write(item["path"] + "\n")
    print(f"✓ Exported file paths to {paths_path}")
    
    # Export VR-only list
    vr_queue = [q for q in queue if q["is_vr"]]
    if vr_queue:
        vr_path = os.path.join(OUTPUT_DIR, "vr_processing_queue.csv")
        with open(vr_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=vr_queue[0].keys())
            writer.writeheader()
            writer.writerows(vr_queue)
        print(f"✓ Exported VR queue ({len(vr_queue)} scenes) to {vr_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING QUEUE SUMMARY")
    print("="*60)
    print(f"\nTotal scenes in queue: {len(queue)}")
    print(f"  - VR scenes: {vr_count}")
    print(f"  - Flat scenes: {flat_count}")
    
    # Count by tier
    local_count = sum(1 for q in queue if q["processing_tier"] == "local")
    runpod_count = sum(1 for q in queue if q["processing_tier"] == "runpod")
    
    print(f"\nBy processing tier:")
    print(f"  - Local (Arc B580): {local_count} scenes")
    print(f"  - RunPod (6K→8K): {runpod_count} scenes")
    
    # Time/cost estimates
    local_hours = local_count * 5  # ~5 hours per scene on Arc B580
    runpod_cost = runpod_count * 3.5  # ~$3.50 per scene
    
    print(f"\nEstimated processing:")
    print(f"  - Local: ~{local_hours} hours ({local_hours/24:.1f} days)")
    print(f"  - RunPod: ~${runpod_cost:.0f}")
    
    print("\n" + "="*60)
    print("FILES CREATED:")
    print("="*60)
    print(f"  1. processing_queue.csv    - Full queue with all details")
    print(f"  2. processing_queue.json   - JSON format for PPP processor")
    print(f"  3. file_paths.txt          - Simple list of file paths")
    print(f"  4. vr_processing_queue.csv - VR scenes only")
    
    # Show first 10 items
    print("\n" + "="*60)
    print("FIRST 10 SCENES IN QUEUE:")
    print("="*60)
    for i, item in enumerate(queue[:10]):
        print(f"  {i+1}. [{item['content_type']:4s}] {item['width']}x{item['height']} → {item['target_resolution']} | {item['title'][:50]}")

if __name__ == "__main__":
    main()
