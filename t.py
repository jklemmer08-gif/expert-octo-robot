import PyNvVideoCodec as nvc
import torch

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

print("format:", f.format)
print("shape:", f.shape)
print("dtype:", f.dtype)

f.nv12_to_rgb()
print("after nv12_to_rgb format:", f.format)
print("after nv12_to_rgb shape:", f.shape)

try:
    img = f.nvcv_image()
    print("nvcv_image type:", type(img))
    print("nvcv_image dir:", [a for a in dir(img) if not a.startswith('_')])
except Exception as e:
    print("nvcv_image error:", e)

try:
    t = torch.from_dlpack(f)
    print("dlpack tensor:", t.shape, t.dtype, t.device)
except Exception as e:
    print("dlpack error:", e)

try:
    plane0 = f.GetPtrToPlane(0)
    print("plane0 type:", type(plane0))
    print("plane0:", plane0)
except Exception as e:
    print("plane0 error:", e)
