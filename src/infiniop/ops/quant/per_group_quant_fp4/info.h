#ifndef __PER_CHANNEL_QUANT_INT8_INFO_H__
#define __PER_CHANNEL_QUANT_INT8_INFO_H__

#include "../../../../utils.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::per_group_quant_fp4 {

class PerGroupQuantF4Info {
private:
    PerGroupQuantF4Info() = default;

public:
    infiniDtype_t input_type, input_global_scale_type, output_scale_type;
    size_t M, N, group_size, output_scale_dimsize;

    static utils::Result<PerGroupQuantF4Info> createPerGroupQuantF4Info(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t output_scale_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t input_global_scale_desc) {

        CHECK_OR_RETURN(
            output_desc != nullptr && output_scale_desc != nullptr && input_desc != nullptr && input_global_scale_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t input_type = input_desc->dtype();
        const infiniDtype_t output_type = output_desc->dtype();
        const infiniDtype_t output_scale_type = output_scale_desc->dtype();
        const infiniDtype_t input_global_scale_type = input_global_scale_desc->dtype();

        CHECK_DTYPE(input_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE(input_global_scale_type, INFINI_DTYPE_F32);
        CHECK_DTYPE(output_type, INFINI_DTYPE_U8);
        CHECK_DTYPE(output_scale_type, INFINI_DTYPE_I32);

        CHECK_OR_RETURN(input_desc->ndim() == 2
                            && output_desc->ndim() == 2
                            && output_scale_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t M = input_desc->dim(0);
        size_t N = input_desc->dim(1);
        size_t group_size = 16;
        if (N % group_size != 0) {
            throw std::runtime_error(
                "The N dimension must be multiple of " + std::to_string(group_size));
        }
        // size_t output_dimsize = N / 2;
        size_t output_scale_dimsize = N / (group_size * 4);

        CHECK_OR_RETURN(M == output_desc->dim(0)
                            || M == output_scale_desc->dim(0)
                            || output_scale_dimsize == output_scale_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<PerGroupQuantF4Info>(PerGroupQuantF4Info{
            input_type,
            input_global_scale_type,
            output_scale_type,
            M,
            N,
            group_size,
            output_scale_dimsize,
        });
    }
};

} // namespace op::per_group_quant_fp4

#endif //  __PER_CHANNEL_QUANT_INT8_INFO_H__
