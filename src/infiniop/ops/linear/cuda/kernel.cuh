#ifndef __LINEAR_KERNEL_CUH__
#define __LINEAR_KERNEL_CUH__

void int8Gemm(
    const int8_t *x_packed, const int8_t *w_packed, int32_t *y_packed,
    int M, int N, int K, cudaStream_t stream) {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    // Use SIMT opclass to avoid tensor-op interleaved layout requirements
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementC, // accumulator type
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm75>;

    Gemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename Gemm::Arguments args{
        problem_size,
        {x_packed, K},
        {w_packed, N},
        {y_packed, N},
        {y_packed, N},
        {1, 0}};

    cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        printf("[CUTLASS SIMT] initialize failed: %d\n", int(status));
        return;
    }
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        printf("[CUTLASS SIMT] run failed: %d\n", int(status));
        return;
    }
}

template <typename Tdata>
__device__ void postKernel(Tdata *y, int32_t *y_packed, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int M, int K, int N, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    int idx = row * N + col;
    float output1 = ((float)x_scale[row] * (float)w_scale[col] * ((float)y_packed[idx] + K * (float)x_zero[row] * (float)w_zero[col]));
    float output2 = 0.0f;
    float output3 = 0.0f;
    float tmp2 = (float)x_scale[row] * (float)w_scale[col] * (float)w_zero[col];
    float tmp3 = (float)x_scale[row] * (float)x_zero[row] * (float)w_scale[col];
    for (int ind = 0; ind < K; ind++) {
        output2 += tmp2 * (float)x_packed[row * K + ind];
        output3 += tmp3 * (float)w_packed[ind * N + col];
    }
    float output = alpha * (output1 - output2 - output3) + beta * (float)c[idx] + (float)bias[col];

    y[idx] = static_cast<Tdata>(output);
}
#endif // __LINEAR_KERNEL_CUH__
