import PyNvVideoCodec as nvc
import torch
import numpy as np

d = nvc.CreateDemuxer("/workspace/test_clip.mp4")
dec = nvc.CreateDecoder(
    gpuid=0,
    codec=d.GetNvCodecId(),
    cudacontext=0,
    cudastream=0
)

it = iter(d)
for _ in range(10):
    frames = dec.Decode(next(it))
    if frames:
        f = frames[0]
        break

# Get NV12 as tensor
nv12 = torch.from_dlpack(f)
print("NV12 shape:", nv12.shape, "device:", nv12.device)

# Move to GPU
nv12 = nv12.cuda()
h = nv12.shape[0] * 2 // 3  # actual height = 1080
w = nv12.shape[1]            # width = 1920

# Split Y and UV planes
y = nv12[:h, :].float()           # (1080, 1920)
uv = nv12[h:, :].float()         # (540, 1920)

# Reshape UV: (540, 1920) -> (540, 960, 2)
uv = uv.reshape(h // 2, w // 2, 2)
u = uv[:, :, 0]  # (540, 960)
v = uv[:, :, 1]  # (540, 960)

# Upsample UV to full resolution
u = u.unsqueeze(0).unsqueeze(0)
v = v.unsqueeze(0).unsqueeze(0)
u = torch.nn.functional.interpolate(u, size=(h, w), mode='nearest').squeeze()
v = torch.nn.functional.interpolate(v, size=(h, w), mode='nearest').squeeze()

# YUV to RGB (BT.601)
r = y + 1.402 * (v - 128)
g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
b = y + 1.772 * (u - 128)

# Stack and clamp
rgb = torch.stack([r, g, b], dim=0).clamp(0, 255).byte()
print("RGB shape:", rgb.shape, "device:", rgb.device)
print("RGB min:", rgb.min().item(), "max:", rgb.max().item())
print("SUCCESS: NV12 -> RGB conversion on GPU")
