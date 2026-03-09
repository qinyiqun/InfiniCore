#include "../../../../devices/nvidia/nvidia_common.cuh"
#include "per_tensor_dequant_fp8_nvidia.cuh"

#include "../../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../../reduce/cuda/reduce.cuh"
#include <cub/block/block_reduce.cuh>

#include "../cuda/kernel.cuh"

template <typename Tin, typename Tout>
INFINIOP_CUDA_KERNEL perTensorDequantFp8Sym(
    Tout *x, const Tin *x_packed, const float *x_scale, int num_elements) {
    perTensorDequantFp8SymKernel<Tin, Tout>(
        x,
        x_packed,
        x_scale,
        num_elements);
}

namespace op::per_tensor_dequant_fp8::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t x_packed_desc,
    infiniopTensorDescriptor_t x_scale_desc,
    infiniopTensorDescriptor_t x_zero_desc) {

    auto info = PerTensorDequantF8Info::createPerTensorDequantF8Info(x_desc, x_packed_desc, x_scale_desc, x_zero_desc);

    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t perTensorDequantFp8(const PerTensorDequantF8Info &info, Tdata *x, const __nv_fp8_e4m3 *x_packed, const float *x_scale, const float *x_zero, cudaStream_t stream) {
    int num_elements = static_cast<int>(info.num_elements);

#ifdef ENABLE_NVIDIA_API
    constexpr unsigned int block_size = 256;
#else
    constexpr unsigned int block_size = BLOCK_SIZE;
#endif
    int num_blocks = min((static_cast<int>(num_elements) + block_size - 1) / block_size, 1024);

    dim3 grid(num_blocks);
    dim3 block(block_size);
    if (x_zero == nullptr) {

        perTensorDequantFp8Sym<__nv_fp8_e4m3, Tdata>
            <<<grid, block, 0, stream>>>(x, x_packed, x_scale, num_elements);
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *x, const void *x_packed, const void *x_scale, const void *x_zero,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define DEQUANT(BLOCK_SIZE, TDATA) \
    perTensorDequantFp8<BLOCK_SIZE, TDATA>(_info, (TDATA *)x, (const __nv_fp8_e4m3 *)x_packed, (const float *)x_scale, (const float *)x_zero, stream)
#define DEQUANT_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                  \
        if (_info.dtype == INFINI_DTYPE_F16)           \
            return DEQUANT(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)      \
            return DEQUANT(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)     \
            return DEQUANT(BLOCK_SIZE, __nv_bfloat16); \
        else                                           \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;     \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        DEQUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        DEQUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        DEQUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::per_tensor_dequant_fp8::nvidia
