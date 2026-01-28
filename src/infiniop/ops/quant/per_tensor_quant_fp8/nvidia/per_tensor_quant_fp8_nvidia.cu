#include "../../../../devices/nvidia/nvidia_common.cuh"
#include "per_tensor_quant_fp8_nvidia.cuh"

#include "../../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../../reduce/cuda/reduce.cuh"
#include <cub/block/block_reduce.cuh>

#include "../cuda/kernel.cuh"

template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockPerTensorAbsmaxSym(
    float *x_scale, const Tdata *x, const int64_t num_elements) {
    per_tensor_absmax_kernel<Tdata, BLOCK_SIZE>(
        x, x_scale, num_elements);
}

template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockPerTensorQuantF8Sym(
    __nv_fp8_e4m3 *x_packed, float *x_scale, const Tdata *x, const int64_t num_elements) {
    per_tensor_quant_fp8_kernel<Tdata, __nv_fp8_e4m3>(
        x,
        x_packed,
        x_scale,
        num_elements);
}

namespace op::per_tensor_quant_fp8::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_packed_desc,
    infiniopTensorDescriptor_t x_scale_desc,
    infiniopTensorDescriptor_t x_zero_desc,
    infiniopTensorDescriptor_t x_desc,
    bool is_static) {

    auto info = PerTensorQuantF8Info::createPerTensorQuantF8Info(x_packed_desc, x_scale_desc, x_zero_desc, x_desc, is_static);

    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t per_tensor_quant_fp8Kernel(const PerTensorQuantF8Info &info, __nv_fp8_e4m3 *x_packed, float *x_scale, float *x_zero, const Tdata *x, cudaStream_t stream) {
    const uint64_t num_elements = info.num_elements;
    bool is_static = info.is_static;
    int num_blocks = (static_cast<int>(num_elements) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (x_zero == nullptr) {
        if (is_static == false) {
            blockPerTensorAbsmaxSym<Tdata, BLOCK_SIZE>
                <<<num_blocks, BLOCK_SIZE, 0, stream>>>(x_scale, x, num_elements);
        }
        blockPerTensorQuantF8Sym<Tdata, BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(x_packed, x_scale, x, num_elements);
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *x_packed, void *x_scale, void *x_zero, const void *x,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define QUANT(BLOCK_SIZE, TDATA) \
    per_tensor_quant_fp8Kernel<BLOCK_SIZE, TDATA>(_info, (__nv_fp8_e4m3 *)x_packed, (float *)x_scale, (float *)x_zero, (const TDATA *)x, stream)
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

} // namespace op::per_tensor_quant_fp8::nvidia
