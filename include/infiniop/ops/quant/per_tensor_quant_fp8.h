#ifndef __INFINIOP_PER_TENSOR_QUANT_FP8_API_H__
#define __INFINIOP_PER_TENSOR_QUANT_FP8_API_H__

#include "../../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopPerTensorQuantF8Descriptor_t;

__C __export infiniStatus_t infiniopCreatePerTensorQuantF8Descriptor(infiniopHandle_t handle,
                                                                     infiniopPerTensorQuantF8Descriptor_t *desc_ptr,
                                                                     infiniopTensorDescriptor_t x_packed_desc,
                                                                     infiniopTensorDescriptor_t x_scale_desc,
                                                                     infiniopTensorDescriptor_t x_zero_desc,
                                                                     infiniopTensorDescriptor_t x_desc,
                                                                     bool is_static);

__C __export infiniStatus_t infiniopGetPerTensorQuantF8WorkspaceSize(infiniopPerTensorQuantF8Descriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopPerTensorQuantF8(infiniopPerTensorQuantF8Descriptor_t desc,
                                                     void *workspace,
                                                     size_t workspace_size,
                                                     void *x_packed,
                                                     void *x_scale,
                                                     void *x_zero,
                                                     const void *x,
                                                     void *stream);

__C __export infiniStatus_t infiniopDestroyPerTensorQuantF8Descriptor(infiniopPerTensorQuantF8Descriptor_t desc);

#endif
