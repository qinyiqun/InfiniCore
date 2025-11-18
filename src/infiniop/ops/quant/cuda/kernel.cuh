#ifndef __QUANT_KERNEL_CUH__
#define __QUANT_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>
// helper: deterministic round-to-even for double
__device__ inline long long round_to_even_double(double x) {
    // epsilon: tolerance for "exactly half"判定 —— 可根据 obs 调整
    const double eps = 1e-12;

    double fl = floor(x);
    double frac = x - fl;

    if (frac > 0.5 + eps) {
        return (long long)fl + 1LL;
    } else if (frac < 0.5 - eps) {
        return (long long)fl;
    } else {
        // frac approximately 0.5 -> ties-to-even
        long long fli = (long long)fl;
        if ((fli & 1LL) == 0LL) {
            return fli; // even -> keep
        } else {
            return fli + 1LL; // odd -> round up
        }
    }
}

template <typename Tdata, unsigned int BLOCK_SIZE>
__device__ void blockQuantKernel(
    int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, const Tdata *x,
    int M, int K) {
    int row = blockIdx.x;
    int tid = row * K;

    // ---- 1. reduce max ----
    float local_max = op::common_cuda::reduce_op::max<BLOCK_SIZE, Tdata>(
        x + tid, K);

    __shared__ float global_max_f;
    if (threadIdx.x == 0) {
        global_max_f = local_max;
    }
    __syncthreads();

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ---- 2. reduce min ----
    float thread_min = __FLT_MAX__;
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE) {
        thread_min = fminf(thread_min, (float)x[tid + ind]);
    }
    float local_min = BlockReduce(temp_storage).Reduce(thread_min, cub::Min());

    __shared__ float global_min_f;
    if (threadIdx.x == 0) {
        global_min_f = local_min;
    }
    __syncthreads();

    // ---- 3. 全部转为 double 精确计算 ----
    double global_max = (double)global_max_f;
    double global_min = (double)global_min_f;

    double scale_d = (global_max - global_min) / 255.0;
    if (scale_d < 1e-8) {
        scale_d = 1e-8;
    }

    double inv_scale_d = 1.0 / scale_d;
    double zero_d = -global_min * inv_scale_d - 128.0;

    // 写回 scale, zero
    x_scale[row] = static_cast<Tdata>(scale_d);
    x_zero[row] = static_cast<Tdata>(zero_d);

    // ---- 4. 使用 double 和 ties-to-even 进行量化 ----
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE) {

        double v = (double)x[tid + ind];
        double qd = v * inv_scale_d + zero_d;

        long long qll = round_to_even_double(qd);

        if (qll > 127LL) {
            qll = 127LL;
        }
        if (qll < -128LL) {
            qll = -128LL;
        }

        x_packed[tid + ind] = static_cast<int8_t>(qll);
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
__device__ void warpQuantKernel(
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

        // use double for scale/zero
        double max_d = (double)max_total[threadIdx.y];
        double min_d = (double)min_total[threadIdx.y];
        double scale_d = (max_d - min_d) / 255.0;
        if (scale_d < 1e-8) {
            scale_d = 1e-8;
        }
        double inv_scale_d = 1.0 / scale_d;
        double zero_d = -min_d * inv_scale_d - 128.0;

        x_scale[otherIdx] = static_cast<Tdata>(scale_d);
        x_zero[otherIdx] = static_cast<Tdata>(zero_d);

        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x) {
            double val = (double)x[tid + ind];
            double qd = val * inv_scale_d + zero_d;
            // nearbyint -> ties-to-even
            long long qll = round_to_even_double(qd); // deterministic ties-to-even

            if (qll > 127LL) {
                qll = 127LL;
            }
            if (qll < -128LL) {
                qll = -128LL;
            }
            x_packed[tid + ind] = static_cast<int8_t>(qll);
        }
    }
}

#endif // __QUANT_KERNEL_CUH__
