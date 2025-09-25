import os
import json
from itertools import product

import torch
import torch.distributed as dist

def measure(m, n, cta, warmup=10, iters=50):
    """
    m: matmul matrix dimension (square: m×m)
    n: comm tensor dimension (square: n×n)
    cta: number of NCCL CTAs to request
    returns dict with matmul_time_ms, comm_time_ms, overlap_time_ms
    """
    dev = torch.device("cuda")

    # force NCCL CTA settings
    os.environ["NCCL_MIN_NCHANNELS"] = str(cta)
    os.environ["NCCL_MAX_NCHANNELS"] = str(cta)
    os.environ["NCCL_MIN_CTAS"]     = str(cta)
    os.environ["NCCL_MAX_CTAS"]     = str(cta)

    # random matmul inputs
    A = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
    B = torch.randn(m, m, device=dev, dtype=torch.bfloat16)

    # random comm buffer, split into a list
    comm     = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    world_sz = dist.get_world_size()

    # chunk into a Python list, not a tuple
    comm_in  = list(comm.chunk(world_sz, dim=0))
    # allocate matching receive buffers
    comm_out = [torch.empty_like(chunk) for chunk in comm_in]

    stream_comp = torch.cuda.Stream()
    stream_comm = torch.cuda.Stream()

    def time_kernel(fn, stream):
        # warmup
        for _ in range(warmup):
            with torch.cuda.stream(stream):
                fn()
        torch.cuda.synchronize()

        # timed
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            with torch.cuda.stream(stream):
                fn()
        end.record(stream)
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    # 1) compute only
    t_mat = time_kernel(lambda: torch.matmul(A, B), stream_comp)

    # 2) comm only
    t_comm = time_kernel(
        lambda: dist.all_to_all(comm_out, comm_in),
        stream_comm
    )

    # 3) overlap both
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        with torch.cuda.stream(stream_comp):
            torch.matmul(A, B)
        with torch.cuda.stream(stream_comm):
            dist.all_to_all(comm_out, comm_in)
    stream_comp.synchronize()
    stream_comm.synchronize()
    end.record()
    torch.cuda.synchronize()
    t_overlap = start.elapsed_time(end) / iters

    return {
        "matmul_ms":   t_mat,
        "comm_ms":     t_comm,
        "overlap_ms":  t_overlap,
        "CTA":         cta,
        "matmul_size": m,
        "comm_size":   n,
        "world_size":  world_sz,
        "rank":        dist.get_rank(),
    }

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    cta_list    = [16, 32, 64]
    sizes_mat   = [2**i for i in range(10, 16)]
    sizes_comm  = [2**i for i in range(10, 16)]

    results = []
    for cta, m_sz, c_sz in product(cta_list, sizes_mat, sizes_comm):
        res = measure(m_sz, c_sz, cta)
        if dist.get_rank() == 0:
            results.append(res)
            print(
                f"CTA={cta:2d} Mat={m_sz:6d}×{m_sz:6d}  "
                f"Comm={c_sz:6d}×{c_sz:6d} →  "
                f"Mat {res['matmul_ms']:.2f}ms  "
                f"Comm {res['comm_ms']:.2f}ms  "
                f"Ovrlp {res['overlap_ms']:.2f}ms"
            )

    if dist.get_rank() == 0:
        with open("overlap_sweep_a2a.json", "w") as f:
            json.dump(results, f, indent=2)

    dist.destroy_process_group()
