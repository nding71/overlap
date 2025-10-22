import os
import json
from itertools import product

import torch
import torch.distributed as dist

def measure(m, k, n, cta, warmup=50, iters=1000):
    """
    GEMM: A(m×k) @ B(k×n) -> C(m×n)
    Comm: AllReduce over an m×k tensor (in-place)
    """
    dev = torch.device("cuda")

    # NOTE: These usually must be set BEFORE init_process_group to take effect.
    os.environ["NCCL_MIN_NCHANNELS"] = str(cta)
    os.environ["NCCL_MAX_NCHANNELS"] = str(cta)
    os.environ["NCCL_MIN_CTAS"]      = str(cta)
    os.environ["NCCL_MAX_CTAS"]      = str(cta)

    # --- random matmul inputs ---
    A = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    B = torch.randn(k, n, device=dev, dtype=torch.bfloat16)

    # --- all_reduce buffer (communicate m × k, in-place) ---
    X = torch.randn(m, k, device=dev, dtype=torch.bfloat16)

    # Reduce op (default SUM). Set REDUCE_OP=MAX/MIN/PROD if desired.
    op_name = os.environ.get("REDUCE_OP", "SUM").upper()
    op = {
        "SUM":  dist.ReduceOp.SUM,
        "PROD": dist.ReduceOp.PRODUCT,
        "MAX":  dist.ReduceOp.MAX,
        "MIN":  dist.ReduceOp.MIN,
    }.get(op_name, dist.ReduceOp.SUM)

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
        return start.elapsed_time(end) / iters  # ms

    # define the collective
    def do_allreduce():
        dist.all_reduce(X, op=op)

    # 1) compute only
    t_mat = time_kernel(lambda: torch.matmul(A, B), stream_comp)

    # 2) comm only (all_reduce of m×k)
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
        "M":           m,
        "K":           k,
        "N":           n,
        "world_size":  dist.get_world_size(),
        "rank":        dist.get_rank(),
        "reduce_op":   op_name,
    }

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")

    # bind to the correct GPU on multi-GPU nodes
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    cta_list = [16, 32, 64]

    # m from 1 to 524,288 doubling each step
    m_list = [2**i for i in range(0, 19)]  # 1 .. 524288
    k_list = [128, 256, 512]
    n_list = [1024, 2048, 4096]

    results = []
    for cta, m_sz, k_sz, n_sz in product(cta_list, m_list, k_list, n_list):
        res = measure(m_sz, k_sz, n_sz, cta)
        if dist.get_rank() == 0:
            results.append(res)
            print(
                f"CTA={cta:2d}  "
                f"M={m_sz:6d}  N={n_sz:6d}  K={k_sz:6d}  "
                f"Mat {res['matmul_ms']:.2f}ms  "
                f"Comm {res['comm_ms']:.2f}ms  "
                f"Ovrlp {res['overlap_ms']:.2f}ms  "
                f"[op={res['reduce_op']}]"
            )

    if dist.get_rank() == 0:
        with open("overlap_sweep_allreduce.json", "w") as f:
            json.dump(results, f, indent=2)

    dist.destroy_process_group()
