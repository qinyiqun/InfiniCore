#ifndef __INFINIOP_PER_TOKEN_QUANT_FP8_API_H__
#define __INFINIOP_PER_TOKEN_QUANT_FP8_API_H__

#include "../../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopPerTokenQuantF8Descriptor_t;

__C __export infiniStatus_t infiniopCreatePerTokenQuantF8Descriptor(infiniopHandle_t handle,
                                                                    infiniopPerTokenQuantF8Descriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t x_packed_desc,
                                                                    infiniopTensorDescriptor_t x_scale_desc,
                                                                    infiniopTensorDescriptor_t x_zero_desc,
                                                                    infiniopTensorDescriptor_t x_desc);

__C __export infiniStatus_t infiniopGetPerTokenQuantF8WorkspaceSize(infiniopPerTokenQuantF8Descriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopPerTokenQuantF8(infiniopPerTokenQuantF8Descriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    void *x_packed,
                                                    void *x_scale,
                                                    void *x_zero,
                                                    const void *x,
                                                    void *stream);

__C __export infiniStatus_t infiniopDestroyPerTokenQuantF8Descriptor(infiniopPerTokenQuantF8Descriptor_t desc);

#endif
