#ifndef __QUANTIZE_W8A8_H__
#define __QUANTIZE_W8A8_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                \
                                                                                             \
    namespace op::quantize_w8a8::NAMESPACE {                                                 \
    class Descriptor final : public InfiniopDescriptor {                                     \
        struct Opaque;                                                                       \
        Opaque *_opaque;                                                                     \
        QuantizeW8a8Info _info;                                                              \
        size_t _workspace_size;                                                              \
                                                                                             \
        Descriptor(Opaque *opaque, QuantizeW8a8Info info,                                    \
                   size_t workspace_size,                                                    \
                   infiniDevice_t device_type, int device_id)                                \
            : InfiniopDescriptor{device_type, device_id},                                    \
              _opaque(opaque), _info(info), _workspace_size(workspace_size) {}               \
                                                                                             \
    public:                                                                                  \
        ~Descriptor();                                                                       \
                                                                                             \
        size_t minWorkspaceSize() const { return _workspace_size; }                          \
                                                                                             \
        static infiniStatus_t create(                                                        \
            infiniopHandle_t handle, Descriptor **desc_ptr,                                  \
            infiniopTensorDescriptor_t c_desc,                                               \
            infiniopTensorDescriptor_t x_desc,                                               \
            infiniopTensorDescriptor_t weights_desc,                                         \
            infiniopTensorDescriptor_t weights_scale_desc,                                   \
            infiniopTensorDescriptor_t weights_zero_desc);                                   \
                                                                                             \
        infiniStatus_t quant(                                                                \
            void *workspace, size_t workspace_size,                                          \
            void *x_packed, void *x_scale, void *x_zero, const void *x, void *stream) const; \
                                                                                             \
        infiniStatus_t calculate(                                                            \
            void *workspace, size_t workspace_size,                                          \
            void *c, void *x_packed,                                                         \
            void *x_scale, void *x_zero, const void *weights,                                \
            const void *weights_scale, const void *weights_zero, void *stream) const;        \
    };                                                                                       \
    }

#endif // __QUANTIZE_W8A8_H__