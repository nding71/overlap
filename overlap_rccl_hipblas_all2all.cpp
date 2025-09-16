#include <mpi.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <rccl/rccl.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#define HIPCHECK(cmd) do { \
  hipError_t e = (cmd); \
  if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while(0)

#define HBLASCHECK(cmd) do { \
  hipblasStatus_t s = (cmd); \
  if (s != HIPBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "hipBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while(0)

#define RCCLCHECK(cmd) do { \
  rcclResult_t r = (cmd); \
  if (r != rcclSuccess) { \
    fprintf(stderr, "RCCL error %s:%d: %s\n", __FILE__, __LINE__, rcclGetErrorString(r)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while(0)

// ---------------- args ----------------
static int getIntArg(int argc, char** argv, const std::string& keyLong, const std::string& keyShort, int defVal) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind(keyLong + "=", 0) == 0) return std::stoi(a.substr(keyLong.size() + 1));
    if (a == keyLong && i + 1 < argc) return std::stoi(argv[i + 1]);
    if (!keyShort.empty() && a == keyShort && i + 1 < argc) return std::stoi(argv[i + 1]);
  }
  return defVal;
}

static void printUsageIfAsked(int argc, char** argv, int rank) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      if (rank == 0) {
        printf(
          "Usage: %s [--M INT] [--N INT] [--K INT] [--batch INT] [--iters INT]\n"
          "Defaults: M=N=K=4096, batch=8, iters=10\n"
          "Example: mpirun -np 4 ./overlap_rccl_hipblas_all2all --M 8192 --N 4096 --K 4096 --batch 8 --iters 20\n",
          argv[0]);
      }
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
  }
}

// ---------------- all-to-all via grouped send/recv ----------------
static inline void rcclAllToAll_grouped(
    const float* sendbuf, float* recvbuf,
    size_t elems_per_peer, rcclComm_t comm, hipStream_t stream, int world_size)
{
  RCCLCHECK(rcclGroupStart());
  for (int p = 0; p < world_size; ++p) {
    const float* s = sendbuf + (size_t)p * elems_per_peer;
    float*       r = recvbuf + (size_t)p * elems_per_peer;
    RCCLCHECK(rcclSend(s, elems_per_peer, rcclFloat, p, comm, stream));
    RCCLCHECK(rcclRecv(r, elems_per_peer, rcclFloat, p, comm, stream));
  }
  RCCLCHECK(rcclGroupEnd());
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_size = 0, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  printUsageIfAsked(argc, argv, world_rank);

  // ---- read args ----
  int M     = getIntArg(argc, argv, "--M",     "-M",     4096);
  int N     = getIntArg(argc, argv, "--N",     "-N",     4096);
  int K     = getIntArg(argc, argv, "--K",     "-K",     4096);
  int batch = getIntArg(argc, argv, "--batch", "-b",     8);
  int iters = getIntArg(argc, argv, "--iters", "-i",     10);
  if (M <= 0 || N <= 0 || K <= 0 || batch <= 0 || iters <= 0) {
    if (world_rank == 0) fprintf(stderr, "All args must be positive.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // device by local rank
  int num_devices = 0;
  HIPCHECK(hipGetDeviceCount(&num_devices));
  int local_rank = 0;
  {
    MPI_Comm shm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm);
    MPI_Comm_rank(shm, &local_rank);
    MPI_Comm_free(&shm);
  }
  HIPCHECK(hipSetDevice(local_rank % num_devices));

  // streams
  hipStream_t stream_compute, stream_comm;
  HIPCHECK(hipStreamCreateWithFlags(&stream_compute, hipStreamNonBlocking));
  HIPCHECK(hipStreamCreateWithFlags(&stream_comm,   hipStreamNonBlocking));

  // hipBLAS
  hipblasHandle_t blas;
  HBLASCHECK(hipblasCreate(&blas));
  HBLASCHECK(hipblasSetStream(blas, stream_compute));

  // RCCL init via MPI
  rcclUniqueId id;
  if (world_rank == 0) RCCLCHECK(rcclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  rcclComm_t comm;
  RCCLCHECK(rcclCommInitRank(&comm, world_size, id, world_rank));

  // BLAS (column-major) layout
  const int lda = M;   // op(A)=N
  const int ldb = K;   // op(B)=N
  const int ldc = M;
  const long long strideA = (long long)lda * K;
  const long long strideB = (long long)ldb * N;
  const long long strideC = (long long)ldc * N;

  // allocate inputs
  float *A=nullptr, *B=nullptr;
  HIPCHECK(hipMalloc(&A, sizeof(float) * strideA * batch));
  HIPCHECK(hipMalloc(&B, sizeof(float) * strideB * batch));

  // ping-pong output buffers (so comm(i) can overlap with gemm(i+1))
  float *C[2] = {nullptr, nullptr};
  HIPCHECK(hipMalloc(&C[0], sizeof(float) * strideC * batch));
  HIPCHECK(hipMalloc(&C[1], sizeof(float) * strideC * batch));

  // all-to-all recv buffers (match total send size)
  const size_t total_elems = (size_t)strideC * (size_t)batch;   // elements
  if (total_elems % (size_t)world_size != 0) {
    if (world_rank == 0)
      fprintf(stderr, "For this simple all-to-all, total_elems (%zu) must be divisible by world_size (%d).\n",
              total_elems, world_size);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const size_t elems_per_peer = total_elems / (size_t)world_size;

  float *A2A_recv[2] = {nullptr, nullptr};
  HIPCHECK(hipMalloc(&A2A_recv[0], sizeof(float) * total_elems));
  HIPCHECK(hipMalloc(&A2A_recv[1], sizeof(float) * total_elems));

  // init
  HIPCHECK(hipMemsetAsync(A, 0x3f, sizeof(float) * strideA * batch, stream_compute));
  HIPCHECK(hipMemsetAsync(B, 0x3f, sizeof(float) * strideB * batch, stream_compute));
  HIPCHECK(hipMemsetAsync(C[0], 0x00, sizeof(float) * strideC * batch, stream_compute));
  HIPCHECK(hipMemsetAsync(C[1], 0x00, sizeof(float) * strideC * batch, stream_compute));
  HIPCHECK(hipStreamSynchronize(stream_compute));

  const float alpha = 1.0f, beta = 0.0f;

  // events for pipelining
  hipEvent_t gemm_done[2];
  HIPCHECK(hipEventCreateWithFlags(&gemm_done[0], hipEventDisableTiming));
  HIPCHECK(hipEventCreateWithFlags(&gemm_done[1], hipEventDisableTiming));

  // ---- prime first GEMM (iter 0) ----
  HBLASCHECK(hipblasSgemmStridedBatched(
      blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
      M, N, K,
      &alpha,
      A, lda, strideA,
      B, ldb, strideB,
      &beta,
      C[0], ldc, strideC,
      batch));
  HIPCHECK(hipEventRecord(gemm_done[0], stream_compute));

  if (world_rank == 0) {
    printf("Config: ranks=%d, M=%d N=%d K=%d, batch=%d, iters=%d, total_elems=%zu, elems_per_peer=%zu\n",
           world_size, M, N, K, batch, iters, total_elems, elems_per_peer);
  }

  // timing (optional)
  hipEvent_t t0, t1;
  HIPCHECK(hipEventCreate(&t0));
  HIPCHECK(hipEventCreate(&t1));
  HIPCHECK(hipEventRecord(t0, nullptr));

  // ---- pipelined loop ----
  for (int i = 1; i < iters; ++i) {
    int cur  = i & 1;      // buffer to compute into
    int prev = cur ^ 1;    // buffer to communicate from

    // (A) kick comm for previous output once its GEMM is done
    HIPCHECK(hipStreamWaitEvent(stream_comm, gemm_done[prev], 0));
    rcclAllToAll_grouped(/*send*/C[prev], /*recv*/A2A_recv[prev],
                         elems_per_peer, comm, stream_comm, world_size);

    // (B) launch next GEMM into cur buffer
    HBLASCHECK(hipblasSgemmStridedBatched(
        blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
        M, N, K,
        &alpha,
        A, lda, strideA,
        B, ldb, strideB,
        &beta,
        C[cur], ldc, strideC,
        batch));
    HIPCHECK(hipEventRecord(gemm_done[cur], stream_compute));
  }

  // drain: run comm for the last produced GEMM buffer (iters-1)
  int last = (iters - 1) & 1;
  HIPCHECK(hipStreamWaitEvent(stream_comm, gemm_done[last], 0));
  rcclAllToAll_grouped(/*send*/C[last], /*recv*/A2A_recv[last],
                       elems_per_peer, comm, stream_comm, world_size);

  // sync
  HIPCHECK(hipStreamSynchronize(stream_compute));
  HIPCHECK(hipStreamSynchronize(stream_comm));
  HIPCHECK(hipEventRecord(t1, nullptr));
  HIPCHECK(hipEventSynchronize(t1));
  float ms = 0.0f;
  HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
  if (world_rank == 0) {
    printf("Pipelined: %d GEMMs + %d AllToAlls in %.3f ms (avg %.3f ms/GEMM+AllToAll)\n",
           iters, iters, ms, ms / iters);
  }

  // cleanup
  HBLASCHECK(hipblasDestroy(blas));
  RCCLCHECK(rcclCommDestroy(comm));
  HIPCHECK(hipFree(A));
  HIPCHECK(hipFree(B));
  HIPCHECK(hipFree(C[0]));
  HIPCHECK(hipFree(C[1]));
  HIPCHECK(hipFree(A2A_recv[0]));
  HIPCHECK(hipFree(A2A_recv[1]));
  HIPCHECK(hipStreamDestroy(stream_compute));
  HIPCHECK(hipStreamDestroy(stream_comm));
  MPI_Finalize();
  return 0;
}
