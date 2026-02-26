#ifndef __PER_TOKEN_DEQUANT_FP8_KERNEL_CUH__
#define __PER_TOKEN_DEQUANT_FP8_KERNEL_CUH__

static constexpr int kWarpSize = 32;

template <typename Tin, typename Tout, unsigned int BLOCK_SIZE>
__device__ void
blockPerTokenDequantFp8SymKernel(Tout *x, const Tin *x_packed, const float *x_scale, int hidden_dim, int num_tokens) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }
    int tid = token_idx * hidden_dim;
    for (int i = threadIdx.x; i < hidden_dim; i += BLOCK_SIZE) {
        float val = static_cast<float>(x_packed[tid + i]) * x_scale[token_idx];
        x[tid + i] = static_cast<Tout>(val);
    }
}

template <typename Tin, typename Tout, unsigned int BLOCK_SIZE, int kTokensPerCTA = 8>
__device__ void
warpPerTokenDequantFp8SymKernel(Tout *x, const Tin *x_packed, const float *x_scale, int hidden_dim, int num_tokens) {
    const int warp_id = threadIdx.x / kWarpSize;       // 0‑7  (8 warps)
    const int lane_id = threadIdx.x & (kWarpSize - 1); // 0‑31
    const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
    if (token_id >= num_tokens) {
        return;
    }
    int tid = token_id * hidden_dim;
    for (int i = lane_id; i < hidden_dim; i += kWarpSize) {
        float val = static_cast<float>(x_packed[tid + i]) * x_scale[token_id];
        x[tid + i] = static_cast<Tout>(val);
    }
}

#endif // __PER_TOKEN_DEQUANT_FP8_KERNEL_CUH__
