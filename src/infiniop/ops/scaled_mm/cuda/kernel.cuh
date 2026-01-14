#ifndef __SCALED_MM_KERNEL_CUH__
#define __SCALED_MM_KERNEL_CUH__

template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const Tdata *bias, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    int idx = row * N + col;
    float output1 = x_scale[row] * w_scale[col] * ((float)y_packed[idx]);

    float output = output1 + (float)bias[col];

    y[idx] = static_cast<Tdata>(output);
}
template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    int idx = row * N + col;
    float output = x_scale[row] * w_scale[col] * ((float)y_packed[idx]);

    y[idx] = static_cast<Tdata>(output);
}
#endif // __SCALED_MM_KERNEL_CUH__
