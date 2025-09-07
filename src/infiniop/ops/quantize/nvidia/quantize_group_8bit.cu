#include <cuda_fp8.h>

#include <cmath>
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "quantize_group_8bit.cuh"
#include "quantize_group_8bit_nvidia.cuh"

namespace op::quantize::nvidia {

struct Descriptor::Opaque {
  std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }


infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t input_desc,
                                  infiniopTensorDescriptor_t output_q_desc,
                                  infiniopTensorDescriptor_t output_s_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = output_q_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_I8, INFINI_DTYPE_F8_E4M3, INFINI_DTYPE_F8_UE8M0);
    auto result =
        QuantizeInfo::create(input_desc, output_q_desc, output_s_desc);

  *desc_ptr = new Descriptor(dtype, result.take(), 0, 
                             new Opaque{handle->internal()}, 
                             handle->device, handle->device_id);
  return INFINI_STATUS_SUCCESS;
}

infiniStatus_t 
Descriptor::calculate(void *workspace, 
                      size_t workspace_size,
                      void *input,
                      void *output_q,
                      void *output_s, 
                      int group_size,
                      double eps, 
                      double min_8bit,
                      double max_8bit, 
                      bool scale_ue8m0,
                      void *stream) const {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);


      //   CHECK_INPUT(input);
  //   CHECK_INPUT(output_q);

  // 需要一个值，要的是input的大小(x * y * z)
  // 需要 out_q.dtype  out_s.stride
  // input.size input.dim
  //
    // const int num_groups = input.numel() / group_size;
    // auto input1 =  _info.input();
    const int num_groups = _info.input()->numel() / group_size;

    // CHECK_EQ(input.numel() % group_size, 0);
    // CHECK_EQ(output_s.dim(), 2);

  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int THREADS_PER_GROUP = 16;

    int groups_per_block = 1;

    if (num_groups % 16 == 0) {
        groups_per_block = 16;
    } else if (num_groups % 8 == 0) {
        groups_per_block = 8;
    } else if (num_groups % 4 == 0) {
        groups_per_block = 4;
    } else if (num_groups % 2 == 0) {
        groups_per_block = 2;
    }

    // auto dst_type = output_q.scalar_type();
    auto dst_type = _info.output_q()->dtype();
    const int num_blocks = num_groups / groups_per_block;
    const int num_threads = groups_per_block * THREADS_PER_GROUP;

    // const bool is_column_major = output_s.stride(0) < output_s.stride(1);
    const bool is_column_major = _info.output_s()->stride(0) < _info.output_s()->stride(1);
    // const int hidden_dim =input.size(input.dim() - 1);
    const int hidden_dim = _info.input()->shape()[_info.input()->ndim() - 1];
    const int num_groups_per_row = hidden_dim / group_size;
    // const int scale_stride = output_s.stride(1);
    const int scale_stride = _info.output_s()->stride(1);
#define LAUNCH_KERNEL(T, DST_DTYPE)                                            \
    do {                                                                         \
    dim3 grid(num_blocks);                                                     \
    dim3 block(num_threads);                                                   \
    if (is_column_major) {                                                     \
        if (scale_ue8m0) {                                                       \
            per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true>            \
                <<<grid, block, 0, cuda_stream>>>(                                      \
                    static_cast<T *>(input), output_q,       \
                    static_cast<uint32_t *>(output_s), group_size,      \
                    num_groups, groups_per_block, (float)eps, (float)min_8bit,     \
                    (float)max_8bit, num_groups_per_row, scale_stride);            \
        } else {                                                                 \
            per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false>           \
                <<<grid, block, 0, cuda_stream>>>(                                      \
                    static_cast<T *>(input), output_q,       \
                    static_cast<float *>(output_s), group_size,         \
                    num_groups, groups_per_block, (float)eps, (float)min_8bit,     \
                    (float)max_8bit, num_groups_per_row, scale_stride);            \
        }                                                                        \
    } else {                                                                   \
        assert(!scale_ue8m0);                                                    \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, false>                   \
            <<<grid, block, 0, cuda_stream>>>(                                        \
                static_cast<T *>(input), output_q,         \
                static_cast<float *>(output_s), group_size,           \
                num_groups, groups_per_block, (float)eps, (float)min_8bit,       \
                (float)max_8bit);                                                \
        }                                                                          \
    } while (0)
    // DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(
    //     input.scalar_type(), scalar_t, [&] {
    //     if (dst_type == at::ScalarType::Char) {
    //         LAUNCH_KERNEL(scalar_t, int8_t);
    //         return true;
    //     } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
    //         LAUNCH_KERNEL(scalar_t, __nv_fp8_e4m3);
    //         return true;
    //     }
    //     return false;
    // });

    if (_info.input()->dtype() == INFINI_DTYPE_F16 && _dtype == INFINI_DTYPE_F8_E4M3)
    switch (_dtype) {
        case INFINI_DTYPE_F8_E4M3:
            LAUNCH_KERNEL(half, __nv_fp8_e4m3);
            // sgl_per_token_group_quant_int8(input, output_q, output_s, group_size, eps, min_8bit, max_8bit, cuda_stream);
            break;
        // case INFINI_DTYPE_F8_E4M3:
        //     // sgl_per_token_group_quant_fp8(input, output_q, output_s, group_size, eps, min_8bit, max_8bit, false, cuda_stream);
        //     break;
        // case INFINI_DTYPE_F8_UE8M0:
            // sgl_per_token_group_quant_fp8(input, output_q, output_s, group_size, eps, min_8bit, max_8bit, true, cuda_stream);
            // break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#undef LAUNCH_KERNEL
    return INFINI_STATUS_SUCCESS;
}

}  // namespace op::dequantize::nvidia