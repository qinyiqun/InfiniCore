#include "../../../operator.h"
#include "../../../handle.h"
#include "infiniop/ops/quant/per_group_quant_fp4.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/per_group_quant_fp4_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreatePerGroupQuantF4Descriptor(infiniopHandle_t handle,
                                                           infiniopPerGroupQuantF4Descriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t output_desc,
                                                           infiniopTensorDescriptor_t output_scale_desc,
                                                           infiniopTensorDescriptor_t input_desc,
                                                           infiniopTensorDescriptor_t input_global_scale_desc) {
#define CREATE(CASE, NAMESPACE)                                                            \
    case CASE:                                                                             \
        return op::per_group_quant_fp4::NAMESPACE::Descriptor::create(                     \
            handle,                                                                        \
            reinterpret_cast<op::per_group_quant_fp4::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                                   \
            output_scale_desc,                                                             \
            input_desc,                                                                    \
            input_global_scale_desc);
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C infiniStatus_t infiniopGetPerGroupQuantF4WorkspaceSize(infiniopPerGroupQuantF4Descriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#define GET(CASE, NAMESPACE)                                                                                  \
    case CASE:                                                                                                \
        *size = reinterpret_cast<op::per_group_quant_fp4::NAMESPACE::Descriptor *>(desc)->minWorkspaceSize(); \
        return INFINI_STATUS_SUCCESS;
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopPerGroupQuantF4(infiniopPerGroupQuantF4Descriptor_t desc,
                                           void *workspace,
                                           size_t workspace_size,
                                           void *output,
                                           void *output_scale,
                                           const void *input,
                                           const void *input_global_scale,
                                           void *stream) {
#define QUANT(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                      \
        return reinterpret_cast<op::per_group_quant_fp4::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, output, output_scale, input, input_global_scale, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        QUANT(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        QUANT(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef QUANT
}

__C infiniStatus_t infiniopDestroyPerGroupQuantF4Descriptor(infiniopPerGroupQuantF4Descriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                         \
    case CASE:                                                                           \
        delete reinterpret_cast<op::per_group_quant_fp4::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
