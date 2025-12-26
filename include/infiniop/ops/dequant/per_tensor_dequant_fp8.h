#ifndef __INFINIOP_PER_TENSOR_DEQUANT_FP8_API_H__
#define __INFINIOP_PER_TENSOR_DEQUANT_FP8_API_H__

#include "../../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopPerTensorDequantF8Descriptor_t;

__C __export infiniStatus_t infiniopCreatePerTensorDequantF8Descriptor(infiniopHandle_t handle,
                                                                       infiniopPerTensorDequantF8Descriptor_t *desc_ptr,
                                                                       infiniopTensorDescriptor_t x_desc,
                                                                       infiniopTensorDescriptor_t x_packed_desc,
                                                                       infiniopTensorDescriptor_t x_scale_desc,
                                                                       infiniopTensorDescriptor_t x_zero_desc);

__C __export infiniStatus_t infiniopGetPerTensorDequantF8WorkspaceSize(infiniopPerTensorDequantF8Descriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopPerTensorDequantF8(infiniopPerTensorDequantF8Descriptor_t desc,
                                                       void *workspace,
                                                       size_t workspace_size,
                                                       void *x,
                                                       const void *x_packed,
                                                       const void *x_scale,
                                                       const void *x_zero,
                                                       void *stream);

__C __export infiniStatus_t infiniopDestroyPerTensorDequantF8Descriptor(infiniopPerTensorDequantF8Descriptor_t desc);

#endif
