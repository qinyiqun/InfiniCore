#include "../../../../devices/nvidia/nvidia_common.cuh"
#include "per_token_quant_fp8_nvidia.cuh"

#include "../../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../../reduce/cuda/reduce.cuh"
#include <cub/block/block_reduce.cuh>

#include "../cuda/kernel.cuh"

template <typename Tdata, typename DST_DTYPE, unsigned int BLOCK_SIZE, int kTokensPerCTA = 8, int kVecSize = 16>
INFINIOP_CUDA_KERNEL warpPerTokenQuantF8Sym(
    DST_DTYPE *x_packed, float *x_scale, const Tdata *x, const int64_t hidden_dim, const int64_t num_tokens) {
    per_token_quant_fp8_kernel<Tdata, DST_DTYPE, BLOCK_SIZE, kTokensPerCTA, kVecSize>(
        x, x_packed, x_scale, hidden_dim, num_tokens);
}

template <typename Tdata, typename DST_DTYPE, unsigned int BLOCK_SIZE, int kVecSize = 16>
INFINIOP_CUDA_KERNEL blockPerTokenQuantF8Sym(
    DST_DTYPE *x_packed, float *x_scale, const Tdata *x, const int64_t hidden_dim, const int64_t num_tokens) {
    per_token_quant_fp8_small_batch_kernel<Tdata, DST_DTYPE, BLOCK_SIZE, kVecSize>(
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

namespace op::per_token_quant_fp8::nvidia {

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
    infiniopTensorDescriptor_t x_desc) {

    auto info = PerTokenQuantF8Info::createPerTokenQuantF8Info(x_packed_desc, x_scale_desc, x_zero_desc, x_desc);

    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t per_token_quant_fp8Kernel(const PerTokenQuantF8Info &info, __nv_fp8_e4m3 *x_packed, float *x_scale, float *x_zero, const Tdata *x, cudaStream_t stream) {

    const int64_t hidden_dim = info.hidden_dim;
    const int64_t num_tokens = info.num_tokens;

    int sm_count = getSMCount();
    const int TOKENS_PER_CTA = 8;
    const bool use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
    const bool use_vec16 = (hidden_dim % 16 == 0);
    const bool use_vec8 = (hidden_dim % 8 == 0);

    if (x_zero == nullptr) {
        if (use_warp_kernel) {

            // -------- warp‑local ---------------------------------------------------
            constexpr int THREADS = TOKENS_PER_CTA * kWarpSize; // 256
            dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
            dim3 block(THREADS);

            if (use_vec16) {
                warpPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, TOKENS_PER_CTA, 16><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            } else if (use_vec8) {
                warpPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, TOKENS_PER_CTA, 8><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            } else {
                warpPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, TOKENS_PER_CTA, 4><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            }
        } else {
            // -------- baseline -----------------------------------------------------
#ifdef ENABLE_NVIDIA_API
            constexpr unsigned int THREADS = 256;
#else
            constexpr unsigned int THREADS = BLOCK_SIZE;
#endif
            dim3 grid(num_tokens);
            dim3 block(THREADS);

            if (use_vec16) {
                blockPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, 16><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            } else if (use_vec8) {
                blockPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, 8><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            } else {
                blockPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, 4><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            }
        }
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
    per_token_quant_fp8Kernel<BLOCK_SIZE, TDATA>(_info, (__nv_fp8_e4m3 *)x_packed, (float *)x_scale, (float *)x_zero, (const TDATA *)x, stream)
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

} // namespace op::per_token_quant_fp8::nvidia
