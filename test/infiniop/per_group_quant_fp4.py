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
_TEST_CASES = [
    # x_shape
    ((128, 64),),
    ((128, 128),),
    ((256, 64),),
    ((256, 128),),
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1.6e-2, "rtol": 1e-5},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


group_size = 16
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0
# E2M1 to float
# 0111 -> 6
# 0110 -> 4
# 0101 -> 3
# 0100 -> 2
# 0011 -> 1.5
# 0010 -> 1
# 0001 -> 0.5
# 0000 -> 0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def cast_from_fp4(x, m, n):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([E2M1_TO_FLOAT32[x] for x in c.flatten()])
    out = out.reshape(m, n).to(torch.float32)
    return out.to(x.device)


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")
    

def per_group_quant_fp4_torch(x, global_scale):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // group_size, group_size))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def recover_swizzled_scales(scale, m, n):
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // group_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]


def test(
    handle,
    device,
    x_shape,
    dtype=InfiniDtype.F16,
    sync=None,
):
    
    print(
        f"Testing Per Group Quant Fp4 on {InfiniDeviceNames[device]} with x_shape:{x_shape}, dtype:{InfiniDtypeNames[dtype]}"
    )
    M, N = x_shape
   
    input = TestTensor(x_shape, None, dtype, device)

    a_global_scale = torch.zeros(1, dtype=torch.float32)
    a_global_scale[0] = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(input.torch_tensor().flatten(), dim=-1)
    ).to(torch.float32)
    
    input_global_scale = TestTensor(
        (1,),
        a_global_scale.stride(),
        InfiniDtype.F32,
        device,
        mode="manual",
        set_tensor=a_global_scale,
    )
    
    output = TestTensor((M, N // 2), None, InfiniDtype.U8, device, mode="zeros")

    rounded_m = ((M + 128 - 1) // 128) * 128
    scale_n = N // group_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4

    output_scale = TestTensor((rounded_m, rounded_n // 4), None, InfiniDtype.I32, device, mode="zeros")

    out_ref, scale_ref = per_group_quant_fp4_torch(input.torch_tensor(), input_global_scale.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePerGroupQuantF4Descriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            output_scale.descriptor,
            input.descriptor,
            input_global_scale.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel

    output.destroy_desc()
    output_scale.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPerGroupQuantF4WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, input.device)

    def lib_per_group_quant_fp4():
        check_error(
            LIBINFINIOP.infiniopPerGroupQuantF4(
                descriptor,
                workspace.data(),
                workspace_size.value,
                output.data(),
                output_scale.data(),
                input.data(),
                input_global_scale.data(),
                None,
            )
        )

    lib_per_group_quant_fp4()
    
    if sync is not None:
        sync()
    
    scale_ans = recover_swizzled_scales(output_scale.actual_tensor().view(torch.float8_e4m3fn), M, N)
    out_ans = cast_from_fp4(output.actual_tensor(), M, N)
    
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(out_ans, out_ref, atol=atol, rtol=rtol)
        debug(scale_ans, scale_ref, atol=atol, rtol=rtol)
    
    assert (torch.allclose(out_ans, out_ref, atol=atol, rtol=rtol) and 
                torch.allclose(scale_ans, scale_ref, atol=atol, rtol=rtol))

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: per_group_quant_fp4_torch(input.torch_tensor(), a_global_scale), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_per_group_quant_fp4(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyPerGroupQuantF4Descriptor(descriptor))


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
