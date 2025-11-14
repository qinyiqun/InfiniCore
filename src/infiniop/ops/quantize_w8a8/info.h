#ifndef __QUANTIZE_W8A8_INFO_H__
#define __QUANTIZE_W8A8_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::quantize_w8a8 {

class QuantizeW8a8Info {
private:
    QuantizeW8a8Info() = default;

public:
    infiniDtype_t dtype, packed_type;
    size_t M, K, N;

    static utils::Result<QuantizeW8a8Info> createQuantizeW8a8Info(
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t weights_desc,
        infiniopTensorDescriptor_t weights_scale_desc,
        infiniopTensorDescriptor_t weights_zero_desc) {

        CHECK_OR_RETURN(
            c_desc != nullptr && x_desc != nullptr && weights_desc != nullptr && weights_scale_desc != nullptr && weights_zero_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t dtype = x_desc->dtype();
        const infiniDtype_t packed_type = weights_desc->dtype();
        CHECK_OR_RETURN(dtype == c_desc->dtype() && dtype == weights_scale_desc->dtype() && dtype == weights_zero_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE(packed_type, INFINI_DTYPE_I8);

        CHECK_OR_RETURN(c_desc->ndim() == 2
                            && x_desc->ndim() == 2
                            && weights_desc->ndim() == 2
                            && weights_scale_desc->ndim() == 2
                            && weights_zero_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t M = c_desc->dim(0);
        size_t N = c_desc->dim(1);
        size_t K = x_desc->dim(1);

        CHECK_OR_RETURN(M == x_desc->dim(0)
                            || K == weights_desc->dim(0)
                            || N == weights_desc->dim(1)
                            || 1 == weights_scale_desc->dim(0)
                            || N == weights_scale_desc->dim(1)
                            || 1 == weights_zero_desc->dim(0)
                            || N == weights_zero_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<QuantizeW8a8Info>(QuantizeW8a8Info{
            dtype,
            packed_type,
            M,
            K,
            N,
        });
    }
};

} // namespace op::quantize_w8a8

#endif //  __QUANTIZE_W8A8_INFO_H__
