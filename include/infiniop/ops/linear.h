#ifndef __INFINIOP_LINEAR_API_H__
#define __INFINIOP_LINEAR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLinearDescriptor_t;

__C __export infiniStatus_t infiniopCreateLinearDescriptor(infiniopHandle_t handle,
                                                           infiniopLinearDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t d_desc,
                                                           infiniopTensorDescriptor_t a_desc,
                                                           infiniopTensorDescriptor_t b_desc,
                                                           infiniopTensorDescriptor_t c_desc);

__C __export infiniStatus_t infiniopGetLinearWorkspaceSize(infiniopLinearDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLinear(infiniopLinearDescriptor_t desc,
                                           void *workspace,
                                           size_t workspace_size,
                                           void *d,
                                           void const *a,
                                           void const *b,
                                           void const *c,
                                           float alpha,
                                           float beta,
                                           void *stream);

__C __export infiniStatus_t infiniopDestroyLinearDescripor(infiniopLinearDescriptor_t desc);

#endif