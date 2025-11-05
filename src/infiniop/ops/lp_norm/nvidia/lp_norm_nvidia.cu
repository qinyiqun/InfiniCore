#include "../../../devices/nvidia/nvidia_common.cuh"
#include "lp_norm_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockLPNorm(
    Tdata *y, const Tdata *x,
    float p,
    size_t dimsize,
    ptrdiff_t stride, float eps) {
    blockLPNormKernel<Tdata, BLOCK_SIZE>(x, y, p, dimsize, stride, eps);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
INFINIOP_CUDA_KERNEL warpLPNorm(
    Tdata *y, const Tdata *x,
    float p,
    size_t othersize,
    size_t dimsize,
    ptrdiff_t stride, float eps) {
    warpLPNormKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x, y, p, othersize, dimsize, stride, eps);
}

namespace op::lp_norm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis,
    int p,
    float eps) {
    auto info = LPNormInfo::createLPNormInfo(y_desc, x_desc, axis, p, eps);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t launchKernel(const LPNormInfo &info, Tdata *y, const Tdata *x,
                            cudaStream_t stream) {
    size_t dimsize = info.dimsize;
    size_t othersize = info.othersize;
    float p_f = static_cast<float>(info.p);
    float eps = info.eps;
    int num_blocks = static_cast<float>(info.othersize);
    ptrdiff_t stride = info.stride;
    if (dimsize > 1024) {
        blockLPNorm<Tdata, BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(y, x,
                                                    p_f, dimsize, stride, eps);
    } else {
        constexpr unsigned int BLOCK_SIZE_x = 32;
        constexpr unsigned int BLOCK_SIZE_y = 32;
        int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
        dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);
        warpLPNorm<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
            <<<grid_dim, block_dim, 0, stream>>>(y, x,
                                                 p_f, othersize, dimsize, stride, eps);
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define CALCULATE_LP_NORM(BLOCK_SIZE, TDATA) \
    launchKernel<BLOCK_SIZE, TDATA>(_info, (TDATA *)y, (const TDATA *)x, stream)
#define CALCULATE_LP_NORM_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                            \
        if (_info.dtype == INFINI_DTYPE_F16)                     \
            return CALCULATE_LP_NORM(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)                \
            return CALCULATE_LP_NORM(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)               \
            return CALCULATE_LP_NORM(BLOCK_SIZE, __nv_bfloat16); \
        else                                                     \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;               \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_LP_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_LP_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_LP_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::lp_norm::nvidia
