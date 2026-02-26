#ifndef __PER_GROUP_QUANR_FP4_H__
#define __PER_GROUP_QUANR_FP4_H__

#include "../../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                         \
                                                                                                                      \
    namespace op::per_group_quant_fp4::NAMESPACE {                                                                    \
    class Descriptor final : public InfiniopDescriptor {                                                              \
        struct Opaque;                                                                                                \
        Opaque *_opaque;                                                                                              \
        PerGroupQuantF4Info _info;                                                                                    \
        size_t _workspace_size;                                                                                       \
                                                                                                                      \
        Descriptor(Opaque *opaque, PerGroupQuantF4Info info,                                                          \
                   size_t workspace_size,                                                                             \
                   infiniDevice_t device_type, int device_id)                                                         \
            : InfiniopDescriptor{device_type, device_id},                                                             \
              _opaque(opaque), _info(info), _workspace_size(workspace_size) {}                                        \
                                                                                                                      \
    public:                                                                                                           \
        ~Descriptor();                                                                                                \
                                                                                                                      \
        size_t minWorkspaceSize() const { return _workspace_size; }                                                   \
                                                                                                                      \
        static infiniStatus_t create(                                                                                 \
            infiniopHandle_t handle, Descriptor **desc_ptr,                                                           \
            infiniopTensorDescriptor_t output_desc,                                                                   \
            infiniopTensorDescriptor_t output_scale_desc,                                                             \
            infiniopTensorDescriptor_t input_desc,                                                                    \
            infiniopTensorDescriptor_t input_global_scale_desc);                                                      \
                                                                                                                      \
        infiniStatus_t calculate(                                                                                     \
            void *workspace, size_t workspace_size,                                                                   \
            void *output, void *output_scale, const void *input, const void *input_global_scale, void *stream) const; \
    };                                                                                                                \
    }

#endif // __PER_GROUP_QUANR_FP4_H__
