#include <mpi.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <rccl/rccl.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <stdexcept>

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

// ----------- simple arg parsing -----------
static int getIntArg(int argc, char** argv, const std::string& keyLong, const std::string& keyShort, int defVal) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    // forms: --key=VAL | --key VAL | -k VAL
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
          "Example: mpirun -np 2 ./overlap_rccl_hipblas --M 8192 --N 8192 --K 4096 --batch 16 --iters 20\n",
          argv[0]);
      }
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_size = 0, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  printUsageIfAsked(argc, argv, world_rank);

  // ---- read args (with sensible defaults) ----
  int M     = getIntArg(argc, argv, "--M",     "-M",     4096);
  int N     = getIntArg(argc, argv, "--N",     "-N",     4096);
  int K     = getIntArg(argc, argv, "--K",     "-K",     4096);
  int batch = getIntArg(argc, argv, "--batch", "-b",     8);
  int iters = getIntArg(argc, argv, "--iters", "-i",     10);

  if (M <= 0 || N <= 0 || K <= 0 || batch <= 0 || iters <= 0) {
    if (world_rank == 0) fprintf(stderr, "All args must be positive.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Pick device by local rank
  int num_devices = 0;
  HIPCHECK(hipGetDeviceCount(&num_devices));

  int local_rank = 0;
  {
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &local_rank);
    MPI_Comm_free(&shmcomm);
  }

  int dev = local_rank % num_devices;
  HIPCHECK(hipSetDevice(dev));

  // Streams
  hipStream_t stream_compute, stream_comm;
  HIPCHECK(hipStreamCreateWithFlags(&stream_compute, hipStreamNonBlocking));
  HIPCHECK(hipStreamCreateWithFlags(&stream_comm, hipStreamNonBlocking));

  // hipBLAS
  hipblasHandle_t blas;
  HBLASCHECK(hipblasCreate(&blas));
  HBLASCHECK(hipblasSetStream(blas, stream_compute));

  // RCCL bootstrap via MPI
  rcclUniqueId id;
  if (world_rank == 0) RCCLCHECK(rcclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  rcclComm_t comm;
  RCCLCHECK(rcclCommInitRank(&comm, world_size, id, world_rank));

  // Strided-batched layout (column-major BLAS math)
  const long long lda = M;  // op(A)=N
  const long long ldb = K;  // op(B)=N
  const long long ldc = M;
  const long long strideA = lda * (long long)K;
  const long long strideB = ldb * (long long)N;
  const long long strideC = ldc * (long long)N;

  // Allocate
  float *A = nullptr, *B = nullptr, *C = nullptr;
  HIPCHECK(hipMalloc(&A, sizeof(float) * strideA * batch));
  HIPCHECK(hipMalloc(&B, sizeof(float) * strideB * batch));
  HIPCHECK(hipMalloc(&C, sizeof(float) * strideC * batch));

  const float alpha = 1.0f, beta = 0.0f;

  // Init
  HIPCHECK(hipMemsetAsync(A, 0x3f, sizeof(float) * strideA * batch, stream_compute));
  HIPCHECK(hipMemsetAsync(B, 0x3f, sizeof(float) * strideB * batch, stream_compute));
  HIPCHECK(hipMemsetAsync(C, 0x00, sizeof(float) * strideC * batch, stream_compute));

  float* comm_buf = C;
  size_t comm_elems = (size_t)(strideC * batch);

  // Warmup (sequential)
  HBLASCHECK(hipblasSgemmStridedBatched(
      blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
      M, N, K,
      &alpha,
      A, (int)lda, strideA,
      B, (int)ldb, strideB,
      &beta,
      C, (int)ldc, strideC,
      batch));
  HIPCHECK(hipStreamSynchronize(stream_compute));

  RCCLCHECK(rcclAllReduce((const void*)comm_buf, (void*)comm_buf, comm_elems,
                          rcclFloat, rcclSum, comm, stream_comm));
  HIPCHECK(hipStreamSynchronize(stream_comm));

  if (world_rank == 0) {
    printf("Config: M=%d N=%d K=%d, batch=%d, iters=%d, ranks=%d\n",
           M, N, K, batch, iters, world_size);
  }

  // Overlap
  hipEvent_t start, stop;
  HIPCHECK(hipEventCreate(&start));
  HIPCHECK(hipEventCreate(&stop));

  HIPCHECK(hipEventRecord(start, nullptr));
  for (int i = 0; i < iters; ++i) {
    HBLASCHECK(hipblasSgemmStridedBatched(
        blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
        M, N, K,
        &alpha,
        A, (int)lda, strideA,
        B, (int)ldb, strideB,
        &beta,
        C, (int)ldc, strideC,
        batch));

    RCCLCHECK(rcclAllReduce((const void*)comm_buf, (void*)comm_buf, comm_elems,
                            rcclFloat, rcclSum, comm, stream_comm));
  }
  HIPCHECK(hipStreamSynchronize(stream_compute));
  HIPCHECK(hipStreamSynchronize(stream_comm));
  HIPCHECK(hipEventRecord(stop, nullptr));
  HIPCHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIPCHECK(hipEventElapsedTime(&ms, start, stop));
  if (world_rank == 0) {
    printf("Completed %d overlapped iterations in %.3f ms (avg %.3f ms/iter)\n",
           iters, ms, ms / iters);
  }

  // Cleanup
  HBLASCHECK(hipblasDestroy(blas));
  RCCLCHECK(rcclCommDestroy(comm));
  HIPCHECK(hipFree(A));
  HIPCHECK(hipFree(B));
  HIPCHECK(hipFree(C));
  HIPCHECK(hipStreamDestroy(stream_compute));
  HIPCHECK(hipStreamDestroy(stream_comm));

  MPI_Finalize();
  return 0;
}
