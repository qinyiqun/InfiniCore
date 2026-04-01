#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor linear_w4a16_gptq(Tensor in, Tensor qweight, Tensor qzeros, Tensor scales, Tensor g_idx);

void linear_w4a16_gptq_(Tensor out, Tensor in, Tensor qweights, Tensor scales, Tensor qzeros, Tensor g_idx);

} // namespace infinicore::op
