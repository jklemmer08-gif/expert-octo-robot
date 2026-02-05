"""Phase 2: VPF Decoder Test"""
import PyNvVideoCodec as nvc
import torch
import numpy as np

print("=" * 60)
print("PHASE 2: VPF Decoder Test")
print("=" * 60)

demuxer = nvc.CreateDemuxer("/workspace/test_clip.mp4")
decoder = nvc.CreateDecoder(
    gpuid=0,
    codec=demuxer.GetNvCodecId(),
    cudacontext=0,
    cudastream=0
)

print(f"Codec: {demuxer.GetNvCodecId()}")

frame_count = 0
for packet in demuxer:
    frames = decoder.decode(packet)
    for frame in frames:
        frame_count += 1
        if frame_count == 1:
            print(f"Frame type: {type(frame)}")
            if hasattr(frame, 'shape'):
                print(f"Frame shape: {frame.shape}")

frames = decoder.decode(None)
for frame in frames:
    frame_count += 1

print(f"Total frames decoded: {frame_count}")
print(f"Expected: 300 (10s @ 30fps)")
if frame_count == 300:
    print("PASSED: VPF decoder works correctly")
else:
    print(f"NOTE: Got {frame_count} frames")
