#ifndef __LP_NORM_KERNEL_CUH__
#define __LP_NORM_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>

template <typename T, unsigned int BLOCK_SIZE>
__device__ void blockLPNormKernel(
    T const *input, T *output, float p, size_t dimsize,
    ptrdiff_t stride, float eps) {

    int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * dimsize; // now, tid = i(JKS) + k(S) + s;

    float p_partial = 0.0f;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        p_partial += powf(input[tid + ind * stride], p);
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float p_total;
    float p_block = BlockReduce(temp_storage).Reduce(p_partial, cub::Sum());
    if (threadIdx.x == 0) { // must set threadIdx.x = 0 write the output to memory
        p_total = powf(p_block, 1.0f / p);
    }
    __syncthreads();
    float inv = __fdividef(1.0F, p_total + eps);

    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE) {
        output[tid + ind * stride] = static_cast<T>(
            static_cast<float>(
                input[tid + ind * stride])
            * inv);
    }
}

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return max(a, b);
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

template <typename T, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpLPNormKernel(T const *input, T *output,
                                 float p, size_t othersize, size_t dimsize,
                                 ptrdiff_t stride, float eps) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;

    if (otherIdx < othersize) {

        __shared__ float p_total[BLOCK_SIZE_y];
        float p_data = 0.0f;

        for (int ph = 0; threadIdx.x + ph * BLOCK_SIZE_x < dimsize; ph++) {
            p_data += powf(static_cast<float>(input[tid + (threadIdx.x + ph * BLOCK_SIZE_x) * stride]), p);
        }

        p_data = WarpAllReduce<SumOp, float, BLOCK_SIZE_x>(p_data);

        if (threadIdx.x == 0) {
            p_total[threadIdx.y] = powf(p_data, 1.0f / p);
        }
        __syncthreads();
        //--------------------------------------------
        float inv = __fdividef(1.0F, p_total[threadIdx.y] + eps);
        for (int ph = 0; threadIdx.x + ph * BLOCK_SIZE_x < dimsize; ph++) {
            output[tid + (threadIdx.x + ph * BLOCK_SIZE_x) * stride] = static_cast<T>(
                static_cast<float>(input[tid + (threadIdx.x + ph * BLOCK_SIZE_x) * stride]) * inv);
        }
    }
}

#endif // __LP_NORM_KERNEL_CUH__
