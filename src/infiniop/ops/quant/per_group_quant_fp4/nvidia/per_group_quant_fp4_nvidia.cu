#include "../../../../devices/nvidia/nvidia_common.cuh"
#include "per_group_quant_fp4_nvidia.cuh"

#include "../../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../../reduce/cuda/reduce.cuh"
#include "../cuda/kernel.cuh"
#include <cub/block/block_reduce.cuh>

#if defined ENABLE_NVIDIA_API
// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
INFINIOP_CUDA_KERNEL
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows, int32_t numCols, Type const *in, float const *SFScale, uint32_t *out, uint32_t *SFout) {
    using PackedVec = PackedVec<Type>;
    static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
    static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

    // Get the global scaling factor, which will be applied to the SF.
    // Note SFScale is the same as next GEMM's alpha, which is
    // (448.f / (Alpha_A / 6.f)).
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

    // Input tensor row/col loops.
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
        for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; colIdx += blockDim.x) {
            int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
            PackedVec in_vec = reinterpret_cast<PackedVec const *>(in)[inOffset];
            // Get the output tensor offset.
            // Same as inOffset because 8 elements are packed into one uint32_t.
            int64_t outOffset = inOffset;
            auto &out_pos = out[outOffset];

            auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

            out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
        }
    }
}

template <typename T>
void invokeFP4Quantization(
    int m,
    int n,
    T const *input,
    float const *SFScale,
    int64_t *output,
    int32_t *SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream) {
    // Grid, Block size.
    // Each thread converts 8 values.
    dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = 2048 / block.x;
    dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

    // Launch the cvt kernel.
    if (useUE8M0) {
        cvt_fp16_to_fp4<T, true><<<grid, block, 0, stream>>>(
            m, n, input, SFScale, reinterpret_cast<uint32_t *>(output), reinterpret_cast<uint32_t *>(SFOuput));
    } else {
        cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
            m, n, input, SFScale, reinterpret_cast<uint32_t *>(output), reinterpret_cast<uint32_t *>(SFOuput));
    }
}

inline int getMultiProcessorCount() {
    static int multi_processor_count = []() {
        int device_id = 0;
        int count = 0;

        // Get the current CUDA device ID
        CHECK_CUDA_SUCCESS(cudaGetDevice(&device_id));

        // Get the number of multiprocessors for the current device
        CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id));

        return count; // Initialize the static variable
    }();

    return multi_processor_count; // Return the cached value on subsequent calls
}

#endif

namespace op::per_group_quant_fp4::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t output_scale_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t input_global_scale_desc) {
    auto info = PerGroupQuantF4Info::createPerGroupQuantF4Info(output_desc, output_scale_desc, input_desc, input_global_scale_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t per_group_quant_fp4Kernel(const PerGroupQuantF4Info &info, int64_t *output, int32_t *output_scale, const Tdata *input, const float *input_global_scale, cudaStream_t stream) {
    int M = (int)info.M;
    int N = (int)info.N;

#if defined ENABLE_NVIDIA_API
    bool useUE8M0 = false;
    int multiProcessorCount = getMultiProcessorCount();
    invokeFP4Quantization(M, N, input, input_global_scale, output, output_scale, useUE8M0, multiProcessorCount, stream);
#endif
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *output,
                                     void *output_scale,
                                     const void *input,
                                     const void *input_global_scale,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define QUANT(BLOCK_SIZE, TDATA) \
    per_group_quant_fp4Kernel<BLOCK_SIZE, TDATA>(_info, (int64_t *)output, (int32_t *)output_scale, (const TDATA *)input, (const float *)input_global_scale, stream)
#define QUANT_WITH_BLOCK_SIZE(BLOCK_SIZE)               \
    {                                                   \
        if (_info.input_type == INFINI_DTYPE_F16)       \
            return QUANT(BLOCK_SIZE, half);             \
        else if (_info.input_type == INFINI_DTYPE_BF16) \
            return QUANT(BLOCK_SIZE, __nv_bfloat16);    \
        else                                            \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;      \
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

} // namespace op::per_group_quant_fp4::nvidia
