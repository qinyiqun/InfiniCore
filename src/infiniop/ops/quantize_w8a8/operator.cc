#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/quantize_w8a8.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/quantize_w8a8_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateQuantizeW8A8Descriptor(infiniopHandle_t handle,
                                                        infiniopQuantizeW8A8Descriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c_desc,
                                                        infiniopTensorDescriptor_t x_desc,
                                                        infiniopTensorDescriptor_t weights_desc,
                                                        infiniopTensorDescriptor_t weights_scale_desc,
                                                        infiniopTensorDescriptor_t weights_zero_desc) {
#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::quantize_w8a8::NAMESPACE::Descriptor::create(                     \
            handle,                                                                  \
            reinterpret_cast<op::quantize_w8a8::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                                  \
            x_desc,                                                                  \
            weights_desc,                                                            \
            weights_scale_desc,                                                      \
            weights_zero_desc);
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

__C infiniStatus_t infiniopGetQuantizeW8A8WorkspaceSize(infiniopQuantizeW8A8Descriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#define GET(CASE, NAMESPACE)                                                                            \
    case CASE:                                                                                          \
        *size = reinterpret_cast<op::quantize_w8a8::NAMESPACE::Descriptor *>(desc)->minWorkspaceSize(); \
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

__C infiniStatus_t infiniopQuantizeW8A8(infiniopQuantizeW8A8Descriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *x_packed,
                                        void *x_scale,
                                        void *x_zero,
                                        const void *x,
                                        void *stream) {
#define QUANT(CASE, NAMESPACE)                                                            \
    case CASE:                                                                            \
        return reinterpret_cast<op::quantize_w8a8::NAMESPACE::Descriptor *>(desc)->quant( \
            workspace, workspace_size, x_packed, x_scale, x_zero, x, stream);

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

__C infiniStatus_t infiniopQuantizeLinearW8A8(infiniopQuantizeW8A8Descriptor_t desc,
                                              void *workspace,
                                              size_t workspace_size,
                                              void *c,
                                              void *x_packed,
                                              void *x_scale,
                                              void *x_zero,
                                              const void *weights,
                                              const void *weights_scale,
                                              const void *weights_zero,
                                              void *stream) {
#define CACULATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                                \
        return reinterpret_cast<op::quantize_w8a8::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, c, x_packed, x_scale, x_zero, weights, weights_scale, weights_zero, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CACULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        CACULATE(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CACULATE
}

__C infiniStatus_t infiniopDestroyQuantizeW8A8Descriptor(infiniopQuantizeW8A8Descriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                   \
    case CASE:                                                                     \
        delete reinterpret_cast<op::quantize_w8a8::NAMESPACE::Descriptor *>(desc); \
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
