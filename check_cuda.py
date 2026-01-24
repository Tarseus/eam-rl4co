import torch, os
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("conda prefix:", os.environ.get("CONDA_PREFIX"))

import torch
from torch.profiler import profile, ProfilerActivity
x = torch.randn(4096,4096, device="cuda")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    y = x @ x
    torch.cuda.synchronize()
print("has cuda time:", any(getattr(e,"self_cuda_time_total",0)>0 for e in prof.key_averages()))

import torch
from torch.profiler import ProfilerActivity, supported_activities
print("supported:", supported_activities())

import os, torch
from pathlib import Path
lib = Path(torch.__file__).parent / "lib" / "libkineto.so"
print("libkineto:", lib)
print("exists:", lib.exists())

import os, torch
base = os.path.join(os.path.dirname(torch.__file__), "lib")
print("torch lib dir:", base)
print("kineto-like files:", [f for f in os.listdir(base) if "kineto" in f.lower()])

import site, os, glob
sp = site.getsitepackages()[0]
print("site-packages:", sp)
cands = glob.glob(os.path.join(sp, "torch*"))
print("torch* entries:")
for p in cands:
    print(" -", os.path.basename(p))