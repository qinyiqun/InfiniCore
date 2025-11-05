#ifndef __LP_NORM_INFO_H__
#define __LP_NORM_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::lp_norm {

class LPNormInfo {
private:
    LPNormInfo() = default;

public:
    //  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
    size_t dimsize;
    size_t othersize;
    ptrdiff_t stride;
    int axis;
    int p;
    float eps;

    //  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<LPNormInfo> createLPNormInfo(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        int axis,
        int p,
        float eps) {
        auto dtype = output_desc->dtype();
        if (dtype != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        auto input_shape = input_desc->shape();

        size_t ndim = input_desc->ndim();

        if (axis < 0) {
            axis += (int)(ndim);
        }
        size_t othersize = 1;
        for (int i = 0; i < (int)ndim; i++) {
            if (i != axis) {
                othersize *= input_shape[i];
            }
        }
        ptrdiff_t stride = 1;
        for (int i = ndim - 1; i > axis; i--) {
            stride *= (ptrdiff_t)input_shape[i];
        }
        size_t dimsize = input_shape[axis];

        return utils::Result<LPNormInfo>(LPNormInfo{
            dtype,
            dimsize,
            othersize,
            stride,
            axis,
            p,
            eps});
    }
};
} // namespace op::lp_norm

#endif //  __LP_NORM_INFO_H__