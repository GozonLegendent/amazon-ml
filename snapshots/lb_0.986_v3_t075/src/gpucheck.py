"""Print how much GPU memory this process can actually use (works without NVML)."""
import os

import torch

print("device:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
free, total = torch.cuda.mem_get_info()
print(f"mem_get_info: free {free / 2**30:.1f} GiB / total {total / 2**30:.1f} GiB")
ok = 0
for g in (2, 4, 8, 16, 24, 32, 40, 60, 75):
    try:
        x = torch.empty(int(g * 2**30), dtype=torch.uint8, device="cuda")
        del x
        torch.cuda.empty_cache()
        ok = g
    except Exception as e:  # noqa: BLE001
        print(f"alloc {g} GiB FAILED: {str(e)[:120]}")
        break
print(f"largest single allocation that worked: {ok} GiB")
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
import time
t = time.time()
for _ in range(20):
    b = a @ a
torch.cuda.synchronize()
print(f"fp16 matmul: {20 * 2 * 8192**3 / (time.time() - t) / 1e12:.0f} TFLOPS")
