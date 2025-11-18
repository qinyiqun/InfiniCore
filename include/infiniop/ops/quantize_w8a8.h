#ifndef __INFINIOP_QUANTIZE_W8A8_API_H__
#define __INFINIOP_QUANTIZE_W8A8_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopQuantizeW8A8Descriptor_t;

__C __export infiniStatus_t infiniopCreateQuantizeW8A8Descriptor(infiniopHandle_t handle,
                                                                 infiniopQuantizeW8A8Descriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t c_desc,
                                                                 infiniopTensorDescriptor_t x_desc,
                                                                 infiniopTensorDescriptor_t weights_desc,
                                                                 infiniopTensorDescriptor_t weights_scale_desc,
                                                                 infiniopTensorDescriptor_t weights_zero_desc);

__C __export infiniStatus_t infiniopGetQuantizeW8A8WorkspaceSize(infiniopQuantizeW8A8Descriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopQuantizeW8A8(infiniopQuantizeW8A8Descriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *x_packed,
                                                 void *x_scale,
                                                 void *x_zero,
                                                 const void *x,
                                                 void *stream);

__C __export infiniStatus_t infiniopQuantizeLinearW8A8(infiniopQuantizeW8A8Descriptor_t desc,
                                                       void *workspace,
                                                       size_t workspace_size,
                                                       void *c,
                                                       void *x_packed,
                                                       void *x_scale,
                                                       void *x_zero,
                                                       const void *weights,
                                                       const void *weights_scale,
                                                       const void *weights_zero,
                                                       void *stream);

__C __export infiniStatus_t infiniopDestroyQuantizeW8A8Descriptor(infiniopQuantizeW8A8Descriptor_t desc);

#endif
