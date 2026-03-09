#ifndef __PER_TENSOR_QUANT_FP8_INFO_H__
#define __PER_TENSOR_QUANT_FP8_INFO_H__

#include "../../../../utils.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::per_tensor_quant_fp8 {

class PerTensorQuantF8Info {
private:
    PerTensorQuantF8Info() = default;

public:
    infiniDtype_t dtype, packed_type;
    int64_t num_elements;
    bool is_static;

    static utils::Result<PerTensorQuantF8Info> createPerTensorQuantF8Info(
        infiniopTensorDescriptor_t x_packed_desc,
        infiniopTensorDescriptor_t x_scale_desc,
        infiniopTensorDescriptor_t x_zero_desc,
        infiniopTensorDescriptor_t x_desc) {

        CHECK_OR_RETURN(
            x_packed_desc != nullptr && x_scale_desc != nullptr && x_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t dtype = x_desc->dtype();
        const infiniDtype_t packed_type = x_packed_desc->dtype();

        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE(packed_type, INFINI_DTYPE_F8);

        auto shape = x_desc->shape();
        CHECK_SAME_SHAPE(shape, x_packed_desc->shape());

        auto ndim = x_desc->ndim();

        int64_t num_elements = 1;
        for (int i = 0; i < (int)ndim; i++) {
            num_elements *= static_cast<int64_t>(shape[i]);
        }

        return utils::Result<PerTensorQuantF8Info>(PerTensorQuantF8Info{
            dtype,
            packed_type,
            num_elements,
        });
    }
};

} // namespace op::per_tensor_quant_fp8

#endif //  __PER_TENSOR_QUANT_FP8_INFO_H__
