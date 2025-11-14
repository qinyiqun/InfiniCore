import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # M, K, N
    (8,8,8),
    (128, 512, 1024),
    (128, 128, 128),
    (256, 1024, 2048),
]

_TEST_CASES = [((M, K), (K, N), (M, N)) for (M, K, N) in _TEST_CASES_]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 5e-2},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 3e-5, "rtol": 5e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def quantize_w8a8(x, w):
    return torch.matmul(x.to(torch.float32), w.to(torch.float32)).to(x.dtype)

def quantWeights(w: torch.Tensor):
    """
    对权重矩阵 w ∈ [K, N] 做 per-channel (按列) 量化。
    返回:
      w_packed: int8 量化权重，形状 [K, N]
      w_scale:  每列的scale，形状 [1, N]，dtype与w相同
      w_zero:   每列的zero point，形状 [1, N]，dtype与w相同
    """
    assert w.dim() == 2, "w must be [K, N]"
    K, N = w.shape

    # 计算每列的最小值和最大值
    w_min = w.min(dim=0, keepdim=True)[0]
    w_max = w.max(dim=0, keepdim=True)[0]

    # 避免除以零
    w_scale = (w_max - w_min) / 255.0
    w_scale = torch.clamp(w_scale, min=1e-8)

    # 计算zero point
    w_zero = -w_min / w_scale - 128.0

    # 计算量化值
    w_q = torch.round(w / w_scale + w_zero)

    # 限制范围[-128, 127]
    w_q = torch.clamp(w_q, -128, 127)

    # 转为int8
    w_packed = w_q.to(torch.int8)

    return w_packed, w_scale.to(w.dtype), w_zero.to(w.dtype)


def test(
    handle,
    device,
    x_shape,
    w_shape,
    y_shape,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing QuantizeW8A8 on {InfiniDeviceNames[device]} with x_shape:{x_shape}, w_shape:{w_shape} dtype:{InfiniDtypeNames[dtype]}"
    )
    M, K = x_shape
    N = w_shape[1]
    x = TestTensor(x_shape, None, dtype, device)
    w = TestTensor(w_shape, None, dtype, device)
    y = TestTensor(y_shape, None, dtype, device)
    ans = quantize_w8a8(x.torch_tensor(), w.torch_tensor())
    x_packed = TestTensor(x_shape, None, InfiniDtype.I8, device, mode="zeros")
    x_scale = TestTensor((M, 1), None, dtype, device)
    x_zero = TestTensor((M, 1), None, dtype, device)

    w_packed, w_scale, w_zero = quantWeights(w.torch_tensor())
    weights = TestTensor(w_shape, None, InfiniDtype.I8, device, mode="manual", set_tensor=w_packed)
    weights_scale = TestTensor((1, N), None, dtype, device, mode="manual", set_tensor=w_scale)
    weights_zero = TestTensor((1, N), None, dtype, device, mode="manual", set_tensor=w_zero)
    
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateQuantizeW8A8Descriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor, weights.descriptor, weights_scale.descriptor, weights_zero.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()
    x_packed.destroy_desc()
    x_scale.destroy_desc()
    x_zero.destroy_desc()
    weights.destroy_desc()
    weights_scale.destroy_desc()
    weights_zero.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetQuantizeW8A8WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_quantize_w8a8():
        check_error(
            LIBINFINIOP.infiniopQuantizeW8A8(
                descriptor,
                workspace.data(),
                workspace_size.value,
                x_packed.data(),
                x_scale.data(),
                x_zero.data(),
                x.data(),
                None,
            )
        )

    lib_quantize_w8a8()

    def lib_quantize_linear_w8a8():
        check_error(
            LIBINFINIOP.infiniopQuantizeLinearW8A8(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x_packed.data(),
                x_scale.data(),
                x_zero.data(),
                weights.data(),
                weights_scale.data(),
                weights_zero.data(),
                None,
            )
        )

    lib_quantize_linear_w8a8()

    if sync is not None:
        sync()
    
    print(max(abs(y.actual_tensor() - ans).flatten()))
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: quantize_w8a8(x.torch_tensor(), w.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_quantize_linear_w8a8(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyQuantizeW8A8Descriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
