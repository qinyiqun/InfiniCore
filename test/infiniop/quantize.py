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
import random

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
torch.manual_seed(43)           # 设置 CPU 种子
torch.cuda.manual_seed(43)      # 设置当前 GPU 种子
torch.cuda.manual_seed_all(43)  # 设置所有 GPU 种子（多卡时）
_TEST_CASES = [
    # alpha, beta, a_shape, b_shape, c_shape, a_stride, b_stride, c_stride
    (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), None, None, None),
    # (1.0, 0.0, (2, 4, 2048), (2, 2048, 2048), (2, 4, 2048), None, None, None),
    # (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), (4096, 1), (4096, 1), (4096, 1)),
    # (1.0, 1.0, (6, 2048), (2048, 2560), (6, 2560), (2048, 1), (1, 2048), (2560, 1)),
    # (1.0 / 8.0, 0.0, (4, 8 * 6, 64), (4, 64, 6), (4, 8 * 6, 6), None, None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    # InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    # InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# PyTorch implementation for matrix multiplication
# def gemm(d, _c, beta, _a, _b, alpha):
#     try:
#         if _c.ndim == 2:
#             torch.addmm(_c, _a, _b, beta=beta, alpha=alpha, out=d)
#         elif _c.ndim == 3:
#             torch.baddbmm(_c, _a, _b, beta=beta, alpha=alpha, out=d)
#         else:
#             raise
#     except Exception:
#         torch.matmul(_a, _b, out=d)
#         d.mul_(alpha).add_(_c, alpha=beta)


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    handle,
    device,
    alpha,
    beta,
    a_shape,
    b_shape,
    c_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Gemm on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
        f" a_shape:{a_shape}, b_shape:{b_shape}, c_shape:{c_shape},"
        f" a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}, dtype:{InfiniDtypeNames[dtype]}"
    )
        
    input = TestTensor((512, 512), None, InfiniDtype.F16, device, mode = "random")
    output_q = TestTensor((512, 512), None, InfiniDtype.F8E4M3, device, mode="zeros")
    outpus_s = TestTensor((5, 512), None, InfiniDtype.F32, device, mode="zeros")
    # out = TestTensor((8192, 2048), None, InfiniDtype.F16, device, mode="zeros")
    
    def quantize_fp8(input_, output_q_, output_s_):
        # 假设原始矩阵形状为 (x, y)
        # x, y = 256, 64  # 举例：x=256，可以被128整除
        # matrix = torch.randn(x, y)  # 随机初始化矩阵

        # 每128行取最大值，得到 (x//128, y) 的矩阵
        print("input:")
        print(input.torch_tensor())
        print("第二列")
        print(input.torch_tensor()[:128,1])
        print(input.torch_tensor()[:128,1].max())
        print(input.torch_tensor()[0,1])
        print(input.torch_tensor()[0,1] / (input.torch_tensor()[:128,1].max() * 448))
        print(input.torch_tensor()[1,1])
        print(input.torch_tensor()[1,1] / (input.torch_tensor()[:128,1].max() * 448))
        print("第三列")
        print(input.torch_tensor()[:128,2])
        print(input.torch_tensor()[:128,2].max())
        print(input.torch_tensor()[0,2])
        print(input.torch_tensor()[0,2] / (input.torch_tensor()[:128,2].max() * 448))
        print(input.torch_tensor()[1,2])
        print(input.torch_tensor()[1,2] / (input.torch_tensor()[:128,2].max() * 448))
        chunk_size = 128
        assert input_.shape[0] % chunk_size == 0, "x 必须能被 128 整除"

        # 方法：reshape 成 (x//128, 128, y)，然后在第1维（128那一维）取最大值
        result = input_.reshape(input_.shape[0] // chunk_size, chunk_size, input_.shape[1]).max(dim=1).values.to(torch.float)
        scale = result / 448
        # print(scale)
        # print(result.shape)
        # print(result)
        max_vals_broadcast = (torch.repeat_interleave(result, chunk_size, dim=0)) 
        
        print("scale")
        print(max_vals_broadcast[:128,1])
        max_vals_broadcast /=448
        print(max_vals_broadcast)

        # result = (input_ / max_vals_broadcast) * 448
        q_val = torch.clamp(input_.to(torch.float) / max_vals_broadcast, -448, 448)
        print('q_val')
        print(q_val[:128, -3])
        # print("q_val")
        # print(q_val.to(torch.float8_e4m3fn))
        return q_val.to(torch.float8_e4m3fn)
        
    ans = quantize_fp8(input.torch_tensor(),None, None)
    
        
    # print(out.actual_tensor())

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateQuantizeDescriptor(
            handle,
            ctypes.byref(descriptor),
            input.descriptor,
            output_q.descriptor,
            outpus_s.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    # for tensor in [a, b, c]:
    #     tensor.destroy_desc()

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetQuantizeWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop gemm operator
    def lib_quantize():
        check_error(
            LIBINFINIOP.infiniopQuantize(
                descriptor,
                workspace.data(),
                workspace_size.value,
                input.data(),
                output_q.data(),
                outpus_s.data(),
                # zeros.data(),
                128,
                0,
                -448,
                448,
                False,
                None,
            )
        )

    lib_quantize()
    
    # print(input.actual_tensor())
    print(ans[:128,-3])
    print(output_q.actual_tensor())
    print(ans)
    # print(outpus_s.actual_tensor())
    
    diff_count = (output_q.actual_tensor() != ans).sum().item()
    print(diff_count)
    # # Validate results
    # atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    # if DEBUG:
    #     debug(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    # assert torch.allclose(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    # # Profiling workflow
    # if PROFILE:
    #     # fmt: off
    #     profile_operation("PyTorch", lambda: torch_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
    #     profile_operation("    lib", lambda: lib_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
    #     # fmt: on
    # check_error(LIBINFINIOP.infiniopDestroyDequantizeDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")