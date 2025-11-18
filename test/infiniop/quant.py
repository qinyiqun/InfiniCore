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
    debug_all,
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
_TEST_CASES = [
    # x_shape, w_shape, y_shape, alpha, beta
    ((8, 8),),
    ((128, 512),),
    ((128, 128),),
    ((256, 1024),),
    ((256, 2048),),
]

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


def quant(w: torch.Tensor):
    
    
    # 计算每列的最小值和最大值
    w_min = w.min(dim=1, keepdim=True)[0]
    w_max = w.max(dim=1, keepdim=True)[0]

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
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Quant on {InfiniDeviceNames[device]} with x_shape:{x_shape}, dtype:{InfiniDtypeNames[dtype]}"
    )
    M, K = x_shape
    x = TestTensor(x_shape, None, dtype, device)
    ans_packed, ans_scale, ans_zero = quant(x.torch_tensor())
    x_packed = TestTensor(x_shape, None, InfiniDtype.I8, device, mode="zeros")
    x_scale = TestTensor((M, 1), None, dtype, device)
    x_zero = TestTensor((M, 1), None, dtype, device)
    
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateQuantDescriptor(
            handle, ctypes.byref(descriptor), x_packed.descriptor, x_scale.descriptor, x_zero.descriptor, x.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    x_packed.destroy_desc()
    x_scale.destroy_desc()
    x_zero.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetQuantWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_quant():
        check_error(
            LIBINFINIOP.infiniopQuant(
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

    lib_quant()

    if sync is not None:
        sync()
    
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype) #quant算子cuda和python难以对齐
    if DEBUG:
        debug_all(
            (x_packed.actual_tensor(), x_scale.actual_tensor(), x_zero.actual_tensor()),
            (ans_packed, ans_scale, ans_zero),
            "and",
            atol=atol,
            rtol=rtol,
        )
    assert (torch.allclose(x_packed.actual_tensor(), ans_packed, atol=atol, rtol=rtol) 
            and torch.allclose(x_scale.actual_tensor(), ans_scale, atol=atol, rtol=rtol) 
            and torch.allclose(x_zero.actual_tensor(), ans_zero, atol=atol, rtol=rtol))

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: quant(x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_quant(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyQuantDescriptor(descriptor))


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
