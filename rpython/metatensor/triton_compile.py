import sys
import triton
from triton.backends.compiler import GPUTarget

ttir, ptx, meta = sys.argv[1:4]
cc = sys.argv[4] if len(sys.argv) > 4 else "auto"
if cc == "auto":
    import torch
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
cc = int(cc)
num_warps = int(sys.argv[5]) if len(sys.argv) > 5 else 4
c = triton.compile(ttir, target=GPUTarget("cuda", cc, 32), options={"num_warps": num_warps})
md = c.metadata
open(ptx, "w").write(c.asm["ptx"])
extra = 2
open(meta, "w").write("%d %d %d\n" % (md.num_warps * 32, md.shared, extra))
