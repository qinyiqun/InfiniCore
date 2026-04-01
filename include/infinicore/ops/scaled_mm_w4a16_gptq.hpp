#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(GptqGemm, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &);

void scaled_mm_w4a16_gptq_(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, const Tensor &g_idx);
} // namespace infinicore::op
