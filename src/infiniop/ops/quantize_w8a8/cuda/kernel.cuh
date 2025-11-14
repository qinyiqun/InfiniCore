#ifndef __QUANTIZE_W8A8_KERNEL_CUH__
#define __QUANTIZE_W8A8_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>

template <typename Tdata, unsigned int BLOCK_SIZE>
__device__ void blockQuantizeKernel(
    int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, const Tdata *x, int M, int K) {
    int tid = blockIdx.x * K;
    float local_max = op::common_cuda::reduce_op::max<BLOCK_SIZE, Tdata>(
        x + tid,
        K);
    __shared__ float global_max;
    if (threadIdx.x == 0) {
        global_max = local_max;
    }
    __syncthreads();
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_min = __FLT_MAX__;
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE) {
        block_min = fminf(block_min, (float)x[tid + ind]);
    }
    float local_min = BlockReduce(temp_storage).Reduce(block_min, cub::Min());
    __shared__ float global_min;
    if (threadIdx.x == 0) {
        global_min = local_min;
    }
    __syncthreads();
    float scale = (global_max - global_min) / 255.0f;
    scale = fmaxf(scale, 1e-8f);
    float zero = -global_min / scale - 128.0f;
    x_scale[blockIdx.x] = static_cast<Tdata>(scale);
    x_zero[blockIdx.x] = static_cast<Tdata>(zero);
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE) {
        float q = roundf((float)x[tid + ind] / scale + zero);
        q = fminf(127.0f, fmaxf(-128.0f, q));
        x_packed[tid + ind] = static_cast<int8_t>(q);
    }
}

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return max(a, b);
    }
};
template <typename T>
struct MinOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return min(a, b);
    }
};
template <template <typename> class ReductionOp, typename T,
          int thread_group_width>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpQuantizeKernel(
    int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, const Tdata *x, int M, int K) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = otherIdx * K;
    if (otherIdx < M) {
        __shared__ float max_total[BLOCK_SIZE_y];
        __shared__ float min_total[BLOCK_SIZE_y];
        float max_data = -__FLT_MAX__;
        float min_data = __FLT_MAX__;
        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x) {
            max_data = fmaxf(max_data, (float)x[tid + ind]);
            min_data = fminf(min_data, (float)x[tid + ind]);
        }
        max_data = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(max_data);
        min_data = WarpAllReduce<MinOp, float, BLOCK_SIZE_x>(min_data);
        if (threadIdx.x == 0) {
            max_total[threadIdx.y] = max_data;
            min_total[threadIdx.y] = min_data;
        }
        __syncthreads();
        float scale = (max_total[threadIdx.y] - min_total[threadIdx.y]) / 255.0f;
        scale = fmaxf(scale, 1e-8f);
        float zero = -min_total[threadIdx.y] / scale - 128.0f;
        x_scale[otherIdx] = static_cast<Tdata>(scale);
        x_zero[otherIdx] = static_cast<Tdata>(zero);
        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x) {
            float q = roundf((float)x[tid + ind] / scale + zero);
            q = fminf(127.0f, fmaxf(-128.0f, q));
            x_packed[tid + ind] = static_cast<int8_t>(q);
        }
    }
}

void int8Gemm(
    int8_t *x_packed, const int8_t *w_packed, int32_t *y_packed,
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
__device__ void postKernel(Tdata *y, int32_t *y_packed, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, int M, int K, int N) {
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
    float output = output1 - output2 - output3;

    y[idx] = static_cast<Tdata>(output);
}
#endif // __QUANTIZE_W8A8_KERNEL_CUH__
