# PPP Library Analysis - Quick Start Guide

## Option 1: Python Script (Recommended)

### Setup

```bash
# On your Linux server
cd /home/jtk1234/media-pipeline/

# Install dependencies
pip install requests pandas tabulate --break-system-packages

# Download the analyzer
# (copy ppp_library_analyzer.py to this directory)

# Edit configuration at top of script:
# - STASH_URL = "http://localhost:9999/graphql"
# - VR_TAG_ID = "299"  # Your VR tag ID
# - OUTPUT_DIR = "./ppp_analysis"
```

### Run Analysis

```bash
python3 ppp_library_analyzer.py
```

### Output Files

After running, you'll have:

```
ppp_analysis/
├── tier1_topaz_candidates.csv    # Your top 50 VR scenes by engagement
├── tier2_runpod_vr.csv           # VR content needing 6K→8K (cloud)
├── tier3_local_bulk.csv          # Everything else for local processing
├── skip_no_upscale_needed.csv    # Already high-res, skip
└── library_summary.json          # Statistics
```

---

## Option 2: Direct Stash GraphQL Query

If you want to quickly pull candidates without running the full script, use this in Stash's GraphQL playground (http://localhost:9999/graphql):

### Find VR Content Under 6K (Tier 2 Candidates)

```graphql
query FindVRUpscaleCandidates {
  findScenes(
    scene_filter: {
      tags: {
        value: ["299"],  # Your VR tag ID
        modifier: INCLUDES
      }
      resolution: {
        modifier: LESS_THAN
        value: FULL_HD  # This catches everything under 6K
      }
    }
    filter: {
      per_page: 100
      sort: "play_count"
      direction: DESC
    }
  ) {
    count
    scenes {
      id
      title
      file {
        path
        width
        height
        bit_rate
        duration
      }
      studio { name }
      rating100
      play_count
      o_counter
    }
  }
}
```

### Find Your Most-Watched Content (Tier 1 Candidates)

```graphql
query FindTopEngagement {
  findScenes(
    scene_filter: {
      o_counter: {
        modifier: GREATER_THAN
        value: 0
      }
    }
    filter: {
      per_page: 50
      sort: "o_counter"
      direction: DESC
    }
  ) {
    count
    scenes {
      id
      title
      file {
        path
        width
        height
      }
      studio { name }
      rating100
      play_count
      o_counter
      tags { name }
    }
  }
}
```

### Find Low-Res 2D Content (Tier 3 Candidates)

```graphql
query FindLowRes2D {
  findScenes(
    scene_filter: {
      tags: {
        value: ["299"],  # VR tag
        modifier: EXCLUDES
      }
      resolution: {
        modifier: LESS_THAN
        value: FOUR_K
      }
    }
    filter: {
      per_page: 200
      sort: "created_at"
      direction: DESC
    }
  ) {
    count
    scenes {
      id
      title
      file {
        path
        width
        height
      }
    }
  }
}
```

---

## Option 3: Quick One-Liner with curl

### Get VR Scene Count by Resolution

```bash
curl -s http://localhost:9999/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ findScenes(scene_filter: {tags: {value: [\"299\"], modifier: INCLUDES}}) { count } }"}' \
  | jq '.data.findScenes.count'
```

### Export VR Scenes to JSON

```bash
curl -s http://localhost:9999/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { findScenes(scene_filter: {tags: {value: [\"299\"], modifier: INCLUDES}}, filter: {per_page: 500}) { scenes { id title file { path width height bit_rate } rating100 o_counter } } }"
  }' | jq '.data.findScenes.scenes' > vr_scenes.json
```

### Quick Stats

```bash
# Count scenes by resolution tier
curl -s http://localhost:9999/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ stats { scene_count } sceneWall { scenes { file { width } } } }"}' \
  | jq '.'
```

---

## Option 4: Stash UI Built-in Filters

You can also do this analysis directly in the Stash UI:

### Finding Tier 1 (Topaz) Candidates:
1. Go to **Scenes**
2. Filter by: **O-Counter > 0** (your favorites)
3. Sort by: **O-Counter DESC**
4. Note the top 20 scene IDs

### Finding Tier 2 (RunPod VR) Candidates:
1. Go to **Scenes**
2. Filter by: **Tags = VR**
3. Filter by: **Resolution < 6K** (or whatever threshold)
4. Sort by: **Rating DESC** or **Play Count DESC**

### Finding Tier 3 (Local Bulk) Candidates:
1. Go to **Scenes**
2. Filter by: **Tags ≠ VR**
3. Filter by: **Resolution < 4K**
4. This is your 2D upscale list

---

## Understanding the Quality Score

The Python script calculates a "quality score" (0-100) for each scene:

| Factor | Points | Logic |
|--------|--------|-------|
| Resolution | 0-25 | Higher res = more points |
| Bitrate | 0-15 | Higher bitrate = better source quality |
| O-Counter | 0-10 | Your explicit preference signal |
| Play Count | 0-5 | Engagement indicator |
| Rating | 0-5 | Your manual rating |

**Score interpretation:**
- 80-100: Definitely process with Topaz (Tier 1)
- 60-80: Strong candidate for cloud processing (Tier 2)
- 40-60: Process locally when time permits (Tier 3)
- <40: Lower priority, process last

---

## Recommended Workflow

### Step 1: Run Initial Analysis
```bash
python3 ppp_library_analyzer.py
```

### Step 2: Review Tier 1 (Manual)
Open `tier1_topaz_candidates.csv` in a spreadsheet:
- Pick your top 10-20 based on personal preference
- These get premium Topaz processing

### Step 3: Confirm Tier 2 (Quick Review)
Open `tier2_runpod_vr.csv`:
- Verify these are actually VR (check filename patterns)
- Remove any misidentified content
- These go to RunPod at ~$3.50 each

### Step 4: Queue Tier 3 (Automated)
`tier3_local_bulk.csv` can be fed directly to your PPP processor:
- Arc B580 processes overnight
- No manual review needed

### Step 5: Skip List (Verify)
Check `skip_no_upscale_needed.csv`:
- These are already 6K+ or don't need upscaling
- Verify nothing was incorrectly skipped

---

## Integration with PPP Processor

Once you have your CSVs, you can feed them to the processing pipeline:

```bash
# Process Tier 3 locally (example)
while IFS=, read -r id title path rest; do
  echo "Queueing: $title"
  curl -X POST http://localhost:8000/jobs \
    -H "Content-Type: application/json" \
    -d "{\"source_path\": \"$path\", \"tier\": \"local\", \"priority\": \"low\"}"
done < <(tail -n +2 tier3_local_bulk.csv | cut -d',' -f1,2,3)
```

Or import directly into the PPP job queue database.

---

## Finding Your VR Tag ID

If you don't know your VR tag ID:

```bash
curl -s http://localhost:9999/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ findTags(tag_filter: {name: {value: \"VR\", modifier: EQUALS}}) { tags { id name } } }"}' \
  | jq '.data.findTags.tags'
```

Or in Stash UI: Tags → VR → look at the URL (the number is the ID)
