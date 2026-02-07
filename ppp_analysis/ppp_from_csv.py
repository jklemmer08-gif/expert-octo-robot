#!/usr/bin/env python3
"""
PPP Queue Generator - From Filtered CSV
Reads your filtered tier3 CSV and generates processing queues.
"""

import csv
import json
import os
from collections import defaultdict

# Input file - update path if needed
INPUT_CSV = "/mnt/user-data/uploads/tier3_filterable_-_Sheet1.csv"
OUTPUT_DIR = "./ppp_queue"

def main():
    print("="*60)
    print("PPP Queue Generator")
    print("="*60)
    
    # Read the CSV
    scenes = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenes.append(row)
    
    print(f"\nLoaded {len(scenes)} scenes from CSV")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Separate by content type
    vr_scenes = [s for s in scenes if s.get('is_vr', '').upper() == 'TRUE']
    flat_scenes = [s for s in scenes if s.get('is_vr', '').upper() != 'TRUE']
    
    print(f"  - VR scenes: {len(vr_scenes)}")
    print(f"  - Flat scenes: {len(flat_scenes)}")
    
    # Get unique performers
    all_performers = set()
    for scene in scenes:
        performers = scene.get('performers', '')
        if performers:
            for p in performers.split(' | '):
                if p.strip():
                    all_performers.add(p.strip())
    
    # Get unique studios
    all_studios = set()
    for scene in scenes:
        studio = scene.get('studio', '')
        if studio:
            all_studios.add(studio)
    
    # Count by resolution
    res_counts = defaultdict(int)
    for scene in scenes:
        tier = scene.get('resolution_tier', 'Unknown')
        res_counts[tier] += 1
    
    # Calculate estimates
    vr_hours = len(vr_scenes) * 5  # ~5 hours each for VR
    flat_hours = len(flat_scenes) * 2  # ~2 hours each for flat 1080p→4K
    total_hours = vr_hours + flat_hours
    
    # Export VR queue
    vr_path = os.path.join(OUTPUT_DIR, "vr_queue.csv")
    if vr_scenes:
        with open(vr_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=vr_scenes[0].keys())
            writer.writeheader()
            writer.writerows(vr_scenes)
        print(f"\n✓ VR queue: {vr_path}")
    
    # Export Flat queue
    flat_path = os.path.join(OUTPUT_DIR, "flat_queue.csv")
    if flat_scenes:
        with open(flat_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=flat_scenes[0].keys())
            writer.writeheader()
            writer.writerows(flat_scenes)
        print(f"✓ Flat queue: {flat_path}")
    
    # Export simple file paths (for batch processing)
    vr_paths = os.path.join(OUTPUT_DIR, "vr_paths.txt")
    with open(vr_paths, 'w') as f:
        for scene in vr_scenes:
            f.write(scene.get('path', '') + '\n')
    print(f"✓ VR paths: {vr_paths}")
    
    flat_paths_file = os.path.join(OUTPUT_DIR, "flat_paths.txt")
    with open(flat_paths_file, 'w') as f:
        for scene in flat_scenes:
            f.write(scene.get('path', '') + '\n')
    print(f"✓ Flat paths: {flat_paths_file}")
    
    # Export JSON for PPP processor
    queue_json = []
    for scene in scenes:
        queue_json.append({
            "id": scene.get('id'),
            "title": scene.get('title', '')[:100],
            "path": scene.get('path', ''),
            "studio": scene.get('studio', ''),
            "performers": scene.get('performers', ''),
            "is_vr": scene.get('is_vr', '').upper() == 'TRUE',
            "width": int(scene.get('width', 0) or 0),
            "height": int(scene.get('height', 0) or 0),
            "resolution_tier": scene.get('resolution_tier', ''),
            "target": scene.get('recommendation', ''),
            "quality_score": float(scene.get('quality_score', 0) or 0),
            "duration_min": float(scene.get('duration_min', 0) or 0),
            "size_gb": float(scene.get('size_gb', 0) or 0),
        })
    
    json_path = os.path.join(OUTPUT_DIR, "processing_queue.json")
    with open(json_path, 'w') as f:
        json.dump(queue_json, f, indent=2)
    print(f"✓ JSON queue: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("QUEUE SUMMARY")
    print("="*60)
    
    print(f"\nTotal scenes: {len(scenes)}")
    print(f"  - VR: {len(vr_scenes)}")
    print(f"  - Flat: {len(flat_scenes)}")
    
    print(f"\nBy resolution:")
    for tier in ['8K+', '6K', '5K', '4K', '1440p', '1080p', '720p', 'SD']:
        count = res_counts.get(tier, 0)
        if count > 0:
            bar = '█' * (count // 5 + 1)
            print(f"  {tier:8s}: {bar} {count}")
    
    print(f"\nUnique performers: {len(all_performers)}")
    print(f"Unique studios: {len(all_studios)}")
    
    # Top studios
    studio_counts = defaultdict(int)
    for scene in scenes:
        studio = scene.get('studio', '')
        if studio:
            studio_counts[studio] += 1
    
    print(f"\nTop 10 studios:")
    for studio, count in sorted(studio_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {studio}: {count}")
    
    # Processing estimates
    print("\n" + "="*60)
    print("PROCESSING ESTIMATES")
    print("="*60)
    
    print(f"\nVR content ({len(vr_scenes)} scenes):")
    print(f"  - Est. time on Arc B580: ~{vr_hours} hours ({vr_hours/24:.1f} days)")
    print(f"  - Processing: 4K/5K → 6K")
    
    print(f"\nFlat content ({len(flat_scenes)} scenes):")
    print(f"  - Est. time on Arc B580: ~{flat_hours} hours ({flat_hours/24:.1f} days)")
    print(f"  - Processing: 1080p → 4K")
    
    print(f"\nTotal estimate: ~{total_hours} hours ({total_hours/24:.1f} days)")
    
    # Storage estimate
    total_current_gb = sum(float(s.get('size_gb', 0) or 0) for s in scenes)
    # Rough estimate: 6K VR ~2x size, 4K flat ~2.5x size
    vr_new_size = sum(float(s.get('size_gb', 0) or 0) for s in vr_scenes) * 2
    flat_new_size = sum(float(s.get('size_gb', 0) or 0) for s in flat_scenes) * 2.5
    
    print(f"\nStorage estimates:")
    print(f"  Current: {total_current_gb:.1f} GB")
    print(f"  After upscale: ~{vr_new_size + flat_new_size:.0f} GB")
    print(f"  Additional needed: ~{(vr_new_size + flat_new_size) - total_current_gb:.0f} GB")
    
    print("\n" + "="*60)
    print("FILES CREATED:")
    print("="*60)
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── vr_queue.csv         ({len(vr_scenes)} VR scenes)")
    print(f"  ├── flat_queue.csv       ({len(flat_scenes)} flat scenes)")
    print(f"  ├── vr_paths.txt         (file paths only)")
    print(f"  ├── flat_paths.txt       (file paths only)")
    print(f"  └── processing_queue.json (for PPP processor)")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("""
1. VR content (priority):
   - Feed vr_paths.txt to Real-ESRGAN batch processor
   - Or use processing_queue.json with PPP processor
   
2. Flat content (lower priority):
   - Process overnight when VR queue is done
   - Consider skipping if storage is tight

3. Quick test first:
   Pick one short VR video and test the full pipeline
   before committing to the full batch.
""")

if __name__ == "__main__":
    main()
