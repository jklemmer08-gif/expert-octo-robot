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
first_frame = None
for packet in demuxer:
    frames = decoder.Decode(packet)
    for frame in frames:
        frame_count += 1
        if frame_count == 1:
            first_frame = frame
            print(f"Frame type: {type(frame)}")
            print(f"Frame shape: {frame.shape}")
            # Explore available methods/attributes
            print(f"Frame attrs: {[a for a in dir(frame) if not a.startswith('_')]}")

# Flush remaining frames from decoder buffer
try:
    # Try creating an empty/end-of-stream packet
    remaining = decoder.Flush()
    for frame in remaining:
        frame_count += 1
except AttributeError:
    pass  # No Flush method, demuxer iteration got all frames

print(f"Total frames decoded: {frame_count}")
print(f"Expected: 300 (10s @ 30fps)")
if frame_count == 300:
    print("PASSED: VPF decoder works correctly")
else:
    print(f"NOTE: Got {frame_count} frames (may differ slightly due to codec buffering)")

# Test converting frame to torch tensor
if first_frame is not None:
    print("\n--- Tensor Conversion Test ---")
    try:
        # Try direct torch tensor conversion via DLPack
        t = torch.from_dlpack(first_frame)
        print(f"DLPack tensor shape: {t.shape}, dtype: {t.dtype}, device: {t.device}")
    except Exception as e1:
        print(f"DLPack failed: {e1}")
        try:
            # Try numpy conversion
            arr = np.array(first_frame)
            t = torch.from_numpy(arr).cuda()
            print(f"NumPy tensor shape: {t.shape}, dtype: {t.dtype}, device: {t.device}")
        except Exception as e2:
            print(f"NumPy failed: {e2}")
            print("Will need custom NV12->RGB conversion")

print("\nPhase 2 complete.")
