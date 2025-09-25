import os
import json
from itertools import product

import torch
import torch.distributed as dist


def measure(m, n, cta, warmup=10, iters=50):
    """
    m: matmul matrix dimension (square: m×m)
    n: comm tensor dimension (square: n×n) for all_reduce
    cta: number of NCCL CTAs (channels) to request
    returns dict with matmul_time_ms, comm_time_ms, overlap_time_ms
    """
    dev = torch.device("cuda")

    # Force NCCL channel/CTA settings (per your sweep)
    os.environ["NCCL_MIN_NCHANNELS"] = str(cta)
    os.environ["NCCL_MAX_NCHANNELS"] = str(cta)
    os.environ["NCCL_MIN_CTAS"] = str(cta)
    os.environ["NCCL_MAX_CTAS"] = str(cta)

    # Random matmul inputs
    A = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
    B = torch.randn(m, m, device=dev, dtype=torch.bfloat16)

    # All-reduce buffer (each rank reduces its own n×n tensor)
    X = torch.randn(n, n, device=dev, dtype=torch.bfloat16)

    # Choose op (default SUM). You can export REDUCE_OP=MAX/MIN/PROD if you want.
    op_name = os.environ.get("REDUCE_OP", "SUM").upper()
    op = {
        "SUM":  dist.ReduceOp.SUM,
        "PROD": dist.ReduceOp.PRODUCT,
        "MAX":  dist.ReduceOp.MAX,
        "MIN":  dist.ReduceOp.MIN,
        # PyTorch also has AVG in recent versions, but NCCL typically does SUM.
    }.get(op_name, dist.ReduceOp.SUM)

    # Streams
    stream_comp = torch.cuda.Stream()
    stream_comm = torch.cuda.Stream()

    # Helper to time a kernel/collective on a given stream
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
        return start.elapsed_time(end) / iters  # milliseconds

    # Define the collective (in-place all_reduce)
    def do_allreduce():
        dist.all_reduce(X, op=op)

    # 1) compute only
    t_mat = time_kernel(lambda: torch.matmul(A, B), stream_comp)

    # 2) comm only (all_reduce)
    t_comm = time_kernel(do_allreduce, stream_comm)

    # 3) overlap both
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        with torch.cuda.stream(stream_comp):
            torch.matmul(A, B)
        with torch.cuda.stream(stream_comm):
            do_allreduce()
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
        "world_size":  dist.get_world_size(),
        "rank":        dist.get_rank(),
        "reduce_op":   op_name,
    }


if __name__ == "__main__":
    # Initialize distributed (use the same launcher you already have)
    dist.init_process_group(backend="nccl")

    # Pin GPU by LOCAL_RANK for multi-GPU nodes
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    cta_list   = [16, 32, 64]
    sizes_mat  = [2**i for i in range(10, 16)]
    sizes_comm = [2**i for i in range(10, 16)]

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
                f"Ovrlp {res['overlap_ms']:.2f}ms  "
                f"[op={res['reduce_op']}]"
            )

    if dist.get_rank() == 0:
        with open("overlap_sweep_allreduce.json", "w") as f:
            json.dump(results, f, indent=2)

    dist.destroy_process_group()
