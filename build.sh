hipcc -O3 overlap_rccl_hipblas_all2all.cpp -o overlap_rccl_hipblas_all2all \
  -lrccl -lhipblas -lmpi

hipcc -O3 overlap_rccl_hipblas_allreduce.cpp -o overlap_rccl_hipblas_allreduce \
  -lrccl -lhipblas -lmpi