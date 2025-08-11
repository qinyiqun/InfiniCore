#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "../../operator.h"
#include "../gemm/info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::linear::NAMESPACE {                            \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        op::gemm::MatmulInfo _info;                              \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            op::gemm::MatmulInfo info,                           \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t d_desc,                   \
            infiniopTensorDescriptor_t a_desc,                   \
            infiniopTensorDescriptor_t b_desc,                   \
            infiniopTensorDescriptor_t c_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            float alpha,                                         \
            const void *a,                                       \
            const void *b,                                       \
            float beta,                                          \
            const void *c,                                       \
            void *d,                                             \
            void *workspace, size_t workspace_size,              \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __LINEAR_H__
