#ifndef __PER_TOKEN_QUANT_FP8_INFO_H__
#define __PER_TOKEN_QUANT_FP8_INFO_H__

#include "../../../../utils.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::per_token_quant_fp8 {

class PerTokenQuantF8Info {
private:
    PerTokenQuantF8Info() = default;

public:
    infiniDtype_t dtype, packed_type;
    int64_t hidden_dim, num_tokens;

    static utils::Result<PerTokenQuantF8Info> createPerTokenQuantF8Info(
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
        CHECK_OR_RETURN(x_desc->ndim() == 2
                            && x_packed_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        auto ndim = x_desc->ndim();

        int64_t num_tokens = static_cast<int64_t>(shape[0]);
        int64_t hidden_dim = static_cast<int64_t>(shape[1]);

        if (hidden_dim % 4 != 0) {
            throw std::runtime_error(
                "Hidden dimension must be divisible by 4, got " + std::to_string(hidden_dim));
        }

        return utils::Result<PerTokenQuantF8Info>(PerTokenQuantF8Info{
            dtype,
            packed_type,
            hidden_dim,
            num_tokens,
        });
    }
};

} // namespace op::per_token_quant_fp8

#endif //  __PER_TOKEN_QUANT_FP8_INFO_H__
