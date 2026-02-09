#include "../../../../devices/nvidia/nvidia_common.cuh"
#include "per_token_dequant_fp8_nvidia.cuh"

#include "../../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../../reduce/cuda/reduce.cuh"
#include <cub/block/block_reduce.cuh>

#include "../cuda/kernel.cuh"

template <typename Tin, typename Tout, unsigned int BLOCK_SIZE, int kTokensPerCTA = 8>
INFINIOP_CUDA_KERNEL warpPerTokenDequantFp8Sym(
    Tout *x, const Tin *x_packed, const float *x_scale, int hidden_dim, int num_tokens) {
    warpPerTokenDequantFp8SymKernel<Tin, Tout, BLOCK_SIZE, kTokensPerCTA>(
        x, x_packed, x_scale, hidden_dim, num_tokens);
}

template <typename Tin, typename Tout, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockPerTokenDequantFp8Sym(
    Tout *x, const Tin *x_packed, const float *x_scale, int hidden_dim, int num_tokens) {
    blockPerTokenDequantFp8SymKernel<Tin, Tout, BLOCK_SIZE>(
        x, x_packed, x_scale, hidden_dim, num_tokens);
}

#ifdef ENABLE_NVIDIA_API
inline int getSMCount() {
    int device = -1;
    cudaGetDevice(&device);

    int sm_count = 0;
    cudaDeviceGetAttribute(
        &sm_count,
        cudaDevAttrMultiProcessorCount,
        device);

    return sm_count;
}

#endif

namespace op::per_token_dequant_fp8::nvidia {

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

    auto info = PerTokenDequantF8Info::createPerTokenDequantF8Info(x_desc, x_packed_desc, x_scale_desc, x_zero_desc);

    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t perTokenDequantFp8(const PerTokenDequantF8Info &info, Tdata *x, const __nv_fp8_e4m3 *x_packed, const float *x_scale, const float *x_zero, cudaStream_t stream) {

    int hidden_dim = static_cast<int>(info.hidden_dim);
    int num_tokens = static_cast<int>(info.num_tokens);

    const int TOKENS_PER_CTA = 8;
#ifdef ENABLE_NVIDIA_API
    int sm_count = getSMCount();
    const bool use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
#else
    const bool use_warp_kernel = (hidden_dim < 1024);
#endif

    if (x_zero == nullptr) {
        if (use_warp_kernel) {

            // -------- warp‑local ---------------------------------------------------
            constexpr int THREADS = TOKENS_PER_CTA * kWarpSize; // 256
            dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
            dim3 block(THREADS);

            warpPerTokenDequantFp8Sym<__nv_fp8_e4m3, Tdata, THREADS, TOKENS_PER_CTA><<<grid, block, 0, stream>>>(
                x,
                x_packed,
                x_scale,
                hidden_dim,
                num_tokens);
        } else {
            // -------- baseline -----------------------------------------------------
#ifdef ENABLE_NVIDIA_API
            constexpr unsigned int THREADS = 256;
#else
            constexpr unsigned int THREADS = BLOCK_SIZE;
#endif
            dim3 grid(num_tokens);
            dim3 block(THREADS);

            blockPerTokenDequantFp8Sym<__nv_fp8_e4m3, Tdata, THREADS><<<grid, block, 0, stream>>>(
                x,
                x_packed,
                x_scale,
                hidden_dim,
                num_tokens);
        }
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
    perTokenDequantFp8<BLOCK_SIZE, TDATA>(_info, (TDATA *)x, (const __nv_fp8_e4m3 *)x_packed, (const float *)x_scale, (const float *)x_zero, stream)
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

} // namespace op::per_token_dequant_fp8::nvidia
