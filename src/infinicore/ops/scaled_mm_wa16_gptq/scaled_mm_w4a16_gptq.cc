#include "infinicore/ops/scaled_mm_w4a16_gptq.hpp"
#include "../../utils.hpp"
#include <iostream>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GptqGemm);

GptqGemm::GptqGemm(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, const Tensor &g_idx) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, in, qweight, scales, qzeros, g_idx);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, in, qweight, scales, qzeros, g_idx);
}
void GptqGemm::execute(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, const Tensor &g_idx) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GptqGemm, out, in, qweight, scales, qzeros, g_idx);
}

void scaled_mm_w4a16_gptq_(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, const Tensor &g_idx) {

    GptqGemm::execute(out, in, qweight, scales, qzeros, g_idx);
}

} // namespace infinicore::op
