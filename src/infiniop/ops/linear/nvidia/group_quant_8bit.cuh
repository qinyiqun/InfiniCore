#include "../../../devices/nvidia/nvidia_common.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

void sgl_per_token_group_quant_int8(
    void* input,
    void* output_q,
    void* output_s,
    int64_t group_size,
    double eps,
    double int8_min,
    double int8_max);

void sgl_per_token_group_quant_fp8(
    void* input,
    void* output_q,
    void* output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0);



infiniStatus_t 
calculate_workspace_size(infiniopTensorDescriptor_t a_desc);