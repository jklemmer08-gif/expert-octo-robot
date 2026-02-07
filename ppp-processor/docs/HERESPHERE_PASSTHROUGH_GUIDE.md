# Heresphere Passthrough Setup Guide

This guide explains how to use processed VR content with Quest 3S passthrough.

## Overview

There are two ways to achieve passthrough background in Heresphere:

1. **Chroma Key (Recommended)** - Real-time keying of green/blue backgrounds
2. **Pre-matted Alpha** - Video with alpha channel (more complex)

## Method 1: Chroma Key (Easy)

### Step 1: Prepare Content

Either:
- Use content that already has solid-color backgrounds
- Process with `matte.py` to add green screen background

```bash
python scripts/matte.py video.mp4 -o video_greenscreen.mp4
```

### Step 2: Configure Heresphere

1. Open video in Heresphere
2. Press **Menu** button
3. Go to **Video** → **Chroma Key**
4. Enable **Chroma Key**
5. Configure settings:

| Setting | Value | Notes |
|---------|-------|-------|
| **Key Color** | #00B140 (RGB 0, 177, 64) | PPP default green |
| **Similarity** | 0.35-0.45 | Adjust based on content |
| **Smoothness** | 0.05-0.15 | Higher = softer edges |
| **Spill Reduction** | 0.1-0.3 | Reduces green tint on edges |

### Step 3: Enable Passthrough

1. Go to **Environment** settings
2. Set **Background** to **Passthrough**
3. Adjust **Passthrough Brightness** as needed

## Method 2: Pre-matted with FFmpeg

For better edge quality, pre-process with alpha:

```bash
# 1. Generate matte with RVM
python scripts/matte.py video.mp4 --output-type alpha_matte

# 2. Combine video with alpha (creates WebM with transparency)
ffmpeg -i video.mp4 -i alpha_matte.mp4 \
  -filter_complex "[0:v][1:v]alphamerge" \
  -c:v libvpx-vp9 -pix_fmt yuva420p \
  output_with_alpha.webm
```

Note: Alpha video support in Heresphere may be limited.

## Optimal Content for Passthrough

### Works Well
- Studio content with clean, solid backgrounds
- POV content (less background visible)
- Content with good lighting separation

### Challenging
- Outdoor scenes with complex backgrounds
- Hair edges (fine detail gets lost)
- Transparent/reflective objects

## Troubleshooting

### Flickering/Unstable Key
- Increase **Smoothness**
- Check source video quality (compression artifacts cause issues)
- Use higher bitrate when processing

### Green Spill on Edges
- Enable **Spill Reduction**
- Process with higher quality model (realesrgan-x4plus)

### Passthrough Too Bright/Dark
- Adjust **Passthrough Brightness** in Environment settings
- Set room lighting appropriately

### Content Looks Flat
- VR projection settings may be wrong
- Check filename has VR identifiers (_180_sbs, etc.)

## Recommended Settings by Content Type

### Studio VR (Indoor)
```
Chroma Key: Enabled
Key Color: Green (#00B140)
Similarity: 0.40
Smoothness: 0.10
Background: Passthrough
```

### POV Content
```
Chroma Key: Enabled
Key Color: Based on background
Similarity: 0.35
Smoothness: 0.15
Background: Passthrough (subtle blend)
```

## File Naming Convention

Ensure files are named correctly for auto-detection:

```
scene_name_180_sbs.mp4       ← 180° side-by-side
scene_name_180x180_LR.mp4    ← 180° left-right
scene_name_360_tb.mp4        ← 360° top-bottom
```

## Quick Test

1. Process a 15-second sample:
   ```bash
   python scripts/upscale.py video.mp4 --sample
   python scripts/matte.py video_sample.mp4 -o test_passthrough.mp4
   ```

2. Transfer to Quest and test in Heresphere

3. Adjust settings before full processing

## Performance Notes

- Chroma keying is GPU-accelerated in Heresphere
- Passthrough adds ~10-15ms latency
- High bitrate videos may stutter with passthrough enabled
- Recommended: 60-80 Mbps for 6K with passthrough
