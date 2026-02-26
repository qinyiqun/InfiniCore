#ifndef __INFINIOP_PER_GROUP_QUANT_FP4_API_H__
#define __INFINIOP_PER_GROUP_QUANT_FP4_API_H__

#include "../../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopPerGroupQuantF4Descriptor_t;

__C __export infiniStatus_t infiniopCreatePerGroupQuantF4Descriptor(infiniopHandle_t handle,
                                                                    infiniopPerGroupQuantF4Descriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t output_desc,
                                                                    infiniopTensorDescriptor_t output_scale_desc,
                                                                    infiniopTensorDescriptor_t input_desc,
                                                                    infiniopTensorDescriptor_t input_global_scale_desc);

__C __export infiniStatus_t infiniopGetPerGroupQuantF4WorkspaceSize(infiniopPerGroupQuantF4Descriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopPerGroupQuantF4(infiniopPerGroupQuantF4Descriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    void *output,
                                                    void *output_scale,
                                                    const void *input,
                                                    const void *input_global_scale,
                                                    void *stream);

__C __export infiniStatus_t infiniopDestroyPerGroupQuantF4Descriptor(infiniopPerGroupQuantF4Descriptor_t desc);

#endif
