#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/reduce_mean.h"

#ifdef ENABLE_CPU_API
#include "cpu/reduce_mean_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/reduce_mean_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/reduce_mean_metax.h"
#endif
// #ifdef ENABLE_ASCEND_API
// #include "ascend/reduce_mean_ascend.h"
// #endif

__C infiniStatus_t infiniopCreateReduceMeanDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMeanDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t dim) {

#define CREATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                     \
        return op::reduce_mean::NAMESPACE::Descriptor::create(                     \
            handle,                                                                \
            reinterpret_cast<op::reduce_mean::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                                \
            x_desc,                                                                \
            dim);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax)
#endif
        // #ifdef ENABLE_ASCEND_API
        //         CREATE(INFINI_DEVICE_ASCEND, ascend)
        // #endif
        // #ifdef ENABLE_CAMBRICON_MLU
        //     case DevCambriconMlu: {
        //         return bangCreateReduceMeanDescriptor((BangHandle_t)handle, (ReduceMeanBangDescriptor_t *)desc_ptr, y_desc);
        //         // return cnnlCreateReduceMeanDescriptor((BangHandle_t) handle, (ReduceMeanCnnlDescriptor_t *) desc_ptr, y_desc);
        //     }
        // #endif
        // #ifdef ENABLE_MTHREADS_GPU
        //     case DevMthreadsGpu: {
        //         return musaCreateReduceMeanDescriptor((MusaHandle_t)handle, (ReduceMeanMusaDescriptor_t *)desc_ptr, y_desc);
        //     }
        // #endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetReduceMeanWorkspaceSize(infiniopReduceMeanDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                     \
        *size = reinterpret_cast<op::reduce_mean::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax)
#endif
        // #ifdef ENABLE_ASCEND_API
        //         GET(INFINI_DEVICE_ASCEND, ascend)
        // #endif
        // #ifdef ENABLE_CAMBRICON_MLU
        //     case DevCambriconMlu: {
        //         return bangGetReduceMeanWorkspaceSize((ReduceMeanBangDescriptor_t)desc, size);
        //         // return cnnlGetReduceMeanWorkspaceSize((ReduceMeanCnnlDescriptor_t) desc, size);
        //     }

        // #endif
        // #ifdef ENABLE_MTHREADS_GPU
        //     case DevMthreadsGpu: {
        //         return musaGetReduceMeanWorkspaceSize((ReduceMeanMusaDescriptor_t)desc, size);
        //     }
        // #endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopReduceMean(
    infiniopReduceMeanDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                              \
        return reinterpret_cast<op::reduce_mean::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax)
#endif
        // #ifdef ENABLE_ASCEND_API
        //         CALCULATE(INFINI_DEVICE_ASCEND, ascend)
        // #endif
        // #ifdef ENABLE_CAMBRICON_MLU
        //     case DevCambriconMlu: {
        //         return bangReduceMean((ReduceMeanBangDescriptor_t)desc, workspace, workspace_size, data, stream);
        //         // return cnnlReduceMean((ReduceMeanCnnlDescriptor_t) desc, workspace, workspace_size, data, stream);
        //     }
        // #endif
        // #ifdef ENABLE_MTHREADS_GPU
        //     case DevMthreadsGpu: {
        //         return musaReduceMean((ReduceMeanMusaDescriptor_t)desc, workspace, workspace_size, data, stream);
        //     }
        // #endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceMeanDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                 \
    case CASE:                                                                   \
        delete reinterpret_cast<op::reduce_mean::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax)
#endif
        // #ifdef ENABLE_ASCEND_API
        //         DESTROY(INFINI_DEVICE_ASCEND, ascend)
        // #endif
        // #ifdef ENABLE_CAMBRICON_MLU
        //     case DevCambriconMlu: {
        //         return bangDestroyReduceMeanDescriptor((ReduceMeanBangDescriptor_t)desc);
        //         // return cnnlDestroyReduceMeanDescriptor((ReduceMeanCnnlDescriptor_t) desc);
        //     }
        // #endif
        // #ifdef ENABLE_MTHREADS_GPU
        //     case DevMthreadsGpu:
        //         return musaDestroyReduceMeanDescriptor((ReduceMeanMusaDescriptor_t)desc);
        // #endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
