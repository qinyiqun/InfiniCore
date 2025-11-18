#include "../../../devices/nvidia/nvidia_common.cuh"
#include "quantize_w8a8_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../reduce/cuda/reduce.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include <cub/block/block_reduce.cuh>

#include "../cuda/kernel.cuh"

template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockQuantize(
    int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, const Tdata *x, int M, int K) {
    blockQuantizeKernel<Tdata, BLOCK_SIZE>(x_packed, x_scale, x_zero, x, M, K);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
INFINIOP_CUDA_KERNEL warpQuantize(
    int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, const Tdata *x, int M, int K) {
    warpQuantizeKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x_packed, x_scale, x_zero, x, M, K);
}

template <typename Tdata>
INFINIOP_CUDA_KERNEL post(
    Tdata *y, int32_t *y_packed, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, int M, int K, int N) {
    postKernel<Tdata>(y, y_packed, w_packed, w_scale, w_zero, x_packed, x_scale, x_zero, M, K, N);
}

namespace op::quantize_w8a8::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t weights_desc,
    infiniopTensorDescriptor_t weights_scale_desc,
    infiniopTensorDescriptor_t weights_zero_desc) {
    auto info = QuantizeW8a8Info::createQuantizeW8a8Info(c_desc, x_desc, weights_desc, weights_scale_desc, weights_zero_desc);
    CHECK_RESULT(info);
    size_t workspace_size = c_desc->dim(0) * c_desc->dim(1) * sizeof(int32_t);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t quantKernel(const QuantizeW8a8Info &info, int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, const Tdata *x, cudaStream_t stream) {
    int M = (int)info.M;
    int K = (int)info.K;

    if (K > 1024) {
        blockQuantize<Tdata, BLOCK_SIZE>
            <<<M, BLOCK_SIZE, 0, stream>>>(x_packed, x_scale, x_zero, x, M, K);
    } else {
        constexpr unsigned int BLOCK_SIZE_x = 32;
        constexpr unsigned int BLOCK_SIZE_y = 32;
        int num_block_x = (M + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
        dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);
        warpQuantize<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
            <<<grid_dim, block_dim, 0, stream>>>(x_packed, x_scale, x_zero, x, M, K);
    }

    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t launchKernel(const QuantizeW8a8Info &info, Tdata *y, int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, const int8_t *weights, const Tdata *weights_scale, const Tdata *weights_zero, cudaStream_t stream, void *workspace) {
    int M = (int)info.M;
    int K = (int)info.K;
    int N = (int)info.N;
    char *workspace_ptr = reinterpret_cast<char *>(workspace);
    int32_t *y_packed = reinterpret_cast<int32_t *>(workspace_ptr);
    constexpr unsigned int BLOCK_SIZE_x = 32;
    constexpr unsigned int BLOCK_SIZE_y = 32;

    int8Gemm(
        x_packed, weights, y_packed,
        M, N, K, stream);

    int num_block_x = (N + BLOCK_SIZE_x - 1) / BLOCK_SIZE_x;
    int num_block_y = (M + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
    dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    post<Tdata><<<grid_dim, block_dim, 0, stream>>>(y, y_packed, weights, weights_scale, weights_zero, x_packed, x_scale, x_zero, M, K, N);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::quant(void *workspace, size_t workspace_size,
                                 void *x_packed, void *x_scale, void *x_zero, const void *x,
                                 void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define QUANT(BLOCK_SIZE, TDATA) \
    quantKernel<BLOCK_SIZE, TDATA>(_info, (int8_t *)x_packed, (TDATA *)x_scale, (TDATA *)x_zero, (const TDATA *)x, stream)
#define QUANT_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                \
        if (_info.dtype == INFINI_DTYPE_F16)         \
            return QUANT(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)    \
            return QUANT(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)   \
            return QUANT(BLOCK_SIZE, __nv_bfloat16); \
        else                                         \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;   \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        QUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        QUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        QUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *c,
                                     void *x_packed,
                                     void *x_scale,
                                     void *x_zero,
                                     const void *weights,
                                     const void *weights_scale,
                                     const void *weights_zero,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define CALCULATE_QUANTIZE_W8A8(BLOCK_SIZE, TDATA) \
    launchKernel<BLOCK_SIZE, TDATA>(_info, (TDATA *)c, (int8_t *)x_packed, (TDATA *)x_scale, (TDATA *)x_zero, (const int8_t *)weights, (const TDATA *)weights_scale, (const TDATA *)weights_zero, stream, workspace)
#define CALCULATE_QUANTIZE_W8A8_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                                  \
        if (_info.dtype == INFINI_DTYPE_F16)                           \
            return CALCULATE_QUANTIZE_W8A8(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)                      \
            return CALCULATE_QUANTIZE_W8A8(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)                     \
            return CALCULATE_QUANTIZE_W8A8(BLOCK_SIZE, __nv_bfloat16); \
        else                                                           \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                     \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_QUANTIZE_W8A8_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_QUANTIZE_W8A8_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_QUANTIZE_W8A8_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::quantize_w8a8::nvidia
