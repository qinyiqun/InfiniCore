#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/scaled_mm_w4a16_gptq.hpp"
#include <infiniop.h>

namespace infinicore::op::scaled_mm_w4a16_gptq_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, GptqGemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, in, qweight, scales, qzeros, g_idx;
};

void *plan(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, const Tensor &g_idx) {
    size_t seed = hash_combine(out, in, qweight, scales, qzeros, g_idx);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, GptqGemm,
        seed,
        out->desc(), in->desc(), qweight->desc(), scales->desc(), qzeros->desc(), g_idx->desc(), false, 4);
    INFINIOP_WORKSPACE_TENSOR(workspace, GptqGemm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(in),
        graph::GraphTensor(qweight),
        graph::GraphTensor(scales),
        graph::GraphTensor(qzeros),
        graph::GraphTensor(g_idx)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopGptqGemm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->in->data(),
        planned->qweight->data(),
        planned->scales->data(),
        planned->qzeros->data(),
        planned->g_idx->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(GptqGemm, &plan, &run, &cleanup);

} // namespace infinicore::op::scaled_mm_w4a16_gptq_impl::infiniop
