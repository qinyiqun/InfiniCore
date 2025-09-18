from .structs import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    infiniopOperatorDescriptor_t,
)

from ctypes import c_int32, c_void_p, c_size_t, POINTER, c_float, c_bool


class OpRegister:
    registry = []

    @classmethod
    def operator(cls, op):
        cls.registry.append(op)
        return op

    @classmethod
    def register_lib(cls, lib):
        for op in cls.registry:
            op(lib)


@OpRegister.operator
def add_(lib):
    lib.infiniopCreateAddDescriptor.restype = c_int32
    lib.infiniopCreateAddDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetAddWorkspaceSize.restype = c_int32
    lib.infiniopGetAddWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAdd.restype = c_int32
    lib.infiniopAdd.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAddDescriptor.restype = c_int32
    lib.infiniopDestroyAddDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def attention_(lib):
    lib.infiniopCreateAttentionDescriptor.restype = c_int32
    lib.infiniopCreateAttentionDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_size_t,
    ]

    lib.infiniopGetAttentionWorkspaceSize.restype = c_int32
    lib.infiniopGetAttentionWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAttention.restype = c_int32
    lib.infiniopAttention.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAttentionDescriptor.restype = c_int32
    lib.infiniopDestroyAttentionDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def causal_softmax_(lib):
    lib.infiniopCreateCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateCausalSoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetCausalSoftmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetCausalSoftmaxWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopCausalSoftmax.restype = c_int32
    lib.infiniopCausalSoftmax.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroyCausalSoftmaxDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def clip_(lib):
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetClipWorkspaceSize.restype = c_int32
    lib.infiniopGetClipWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def conv_(lib):
    pass


@OpRegister.operator
def gemm_(lib):
    lib.infiniopCreateGemmDescriptor.restype = c_int32
    lib.infiniopCreateGemmDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetGemmWorkspaceSize.restype = c_int32
    lib.infiniopGetGemmWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopGemm.restype = c_int32
    lib.infiniopGemm.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_float,
        c_float,
        c_void_p,
    ]

    lib.infiniopDestroyGemmDescriptor.restype = c_int32
    lib.infiniopDestroyGemmDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def mul_(lib):
    lib.infiniopCreateMulDescriptor.restype = c_int32
    lib.infiniopCreateMulDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetMulWorkspaceSize.restype = c_int32
    lib.infiniopGetMulWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopMul.restype = c_int32
    lib.infiniopMul.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyMulDescriptor.restype = c_int32
    lib.infiniopDestroyMulDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def random_sample_(lib):
    lib.infiniopCreateRandomSampleDescriptor.restype = c_int32
    lib.infiniopCreateRandomSampleDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetRandomSampleWorkspaceSize.restype = c_int32
    lib.infiniopGetRandomSampleWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRandomSample.restype = c_int32
    lib.infiniopRandomSample.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_size_t,
        c_void_p,
        c_float,
        c_float,
        c_int32,
        c_float,
        c_void_p,
    ]

    lib.infiniopDestroyRandomSampleDescriptor.restype = c_int32
    lib.infiniopDestroyRandomSampleDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def rearrange_(lib):
    lib.infiniopCreateRearrangeDescriptor.restype = c_int32
    lib.infiniopCreateRearrangeDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopRearrange.restype = c_int32
    lib.infiniopRearrange.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRearrangeDescriptor.restype = c_int32
    lib.infiniopDestroyRearrangeDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def relu_(lib):
    lib.infiniopCreateReluDescriptor.restype = c_int32
    lib.infiniopCreateReluDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopRelu.restype = c_int32
    lib.infiniopRelu.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyReluDescriptor.restype = c_int32
    lib.infiniopDestroyReluDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def rms_norm_(lib):
    lib.infiniopCreateRMSNormDescriptor.restype = c_int32
    lib.infiniopCreateRMSNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]

    lib.infiniopGetRMSNormWorkspaceSize.restype = c_int32
    lib.infiniopGetRMSNormWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRMSNorm.restype = c_int32
    lib.infiniopRMSNorm.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRMSNormDescriptor.restype = c_int32
    lib.infiniopDestroyRMSNormDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def rope_(lib):
    lib.infiniopCreateRoPEDescriptor.restype = c_int32
    lib.infiniopCreateRoPEDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]

    lib.infiniopGetRoPEWorkspaceSize.restype = c_int32
    lib.infiniopGetRoPEWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRoPE.restype = c_int32
    lib.infiniopRoPE.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRoPEDescriptor.restype = c_int32
    lib.infiniopDestroyRoPEDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def sub_(lib):
    lib.infiniopCreateSubDescriptor.restype = c_int32
    lib.infiniopCreateSubDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSubWorkspaceSize.restype = c_int32
    lib.infiniopGetSubWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopSub.restype = c_int32
    lib.infiniopSub.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySubDescriptor.restype = c_int32
    lib.infiniopDestroySubDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def swiglu_(lib):
    lib.infiniopCreateSwiGLUDescriptor.restype = c_int32
    lib.infiniopCreateSwiGLUDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSwiGLUWorkspaceSize.restype = c_int32
    lib.infiniopGetSwiGLUWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopSwiGLU.restype = c_int32
    lib.infiniopSwiGLU.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySwiGLUDescriptor.restype = c_int32
    lib.infiniopDestroySwiGLUDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def conv_(lib):
    lib.infiniopCreateConvDescriptor.restype = c_int32
    lib.infiniopCreateConvDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_size_t,
    ]
    lib.infiniopGetConvWorkspaceSize.restype = c_int32
    lib.infiniopGetConvWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopConv.restype = c_int32
    lib.infiniopConv.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyConvDescriptor.restype = c_int32
    lib.infiniopDestroyConvDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def topkrouter_(lib):
    lib.infiniopCreateTopkrouterDescriptor.restype = c_int32
    lib.infiniopCreateTopkrouterDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetTopkrouterWorkspaceSize.restype = c_int32
    lib.infiniopGetTopkrouterWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopTopkrouter.restype = c_int32
    lib.infiniopTopkrouter.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_float,
        c_size_t,
        c_void_p,
    ]

    lib.infiniopDestroyTopkrouterDescriptor.restype = c_int32
    lib.infiniopDestroyTopkrouterDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def dequantize_(lib):
    lib.infiniopCreateDequantizeDescriptor.restype = c_int32
    lib.infiniopCreateDequantizeDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetDequantizeWorkspaceSize.restype = c_int32
    lib.infiniopGetDequantizeWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopDequantize.restype = c_int32
    lib.infiniopDequantize.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_size_t,
        c_size_t,
        c_size_t,
        c_void_p,
    ]
    lib.infiniopDestroyDequantizeDescriptor.restype = c_int32
    lib.infiniopDestroyDequantizeDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]

@OpRegister.operator
def softplus_(lib):
    lib.infiniopCreateSoftplusDescriptor.restype = c_int32
    lib.infiniopCreateSoftplusDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopSoftplus.restype = c_int32
    lib.infiniopSoftplus.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroySoftplusDescriptor.restype = c_int32
    lib.infiniopDestroySoftplusDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def cross_entropy_loss_(lib):
    lib.infiniopCreateCrossEntropyLossDescriptor.restype = c_int32
    lib.infiniopCreateCrossEntropyLossDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # loss_desc
        infiniopTensorDescriptor_t,  # logits_desc
        infiniopTensorDescriptor_t,  # target_desc
    ]

    lib.infiniopGetCrossEntropyLossWorkspaceSize.restype = c_int32
    lib.infiniopGetCrossEntropyLossWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopCrossEntropyLoss.restype = c_int32
    lib.infiniopCrossEntropyLoss.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # loss
        c_void_p,  # logits
        c_void_p,  # target
        c_void_p,  # stream
    ]

    lib.infiniopDestroyCrossEntropyLossDescriptor.restype = c_int32
    lib.infiniopDestroyCrossEntropyLossDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def avg_pool_(lib):
    lib.infiniopCreateAvgPoolDescriptor.restype = c_int32
    lib.infiniopCreateAvgPoolDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # output_desc
        infiniopTensorDescriptor_t,  # input_desc
        c_void_p,  # kernel_size
        c_void_p,  # strides
        c_void_p,  # pads
        c_bool,  # ceil_mode
    ]


    lib.infiniopGetAvgPoolWorkspaceSize.restype = c_int32
    lib.infiniopGetAvgPoolWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopAvgPool.restype = c_int32
    lib.infiniopAvgPool.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # output
        c_void_p,  # input
        c_void_p,  # stream
    ]

    lib.infiniopDestroyAvgPoolDescriptor.restype = c_int32
    lib.infiniopDestroyAvgPoolDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def exp_(lib):
    lib.infiniopCreateExpDescriptor.restype = c_int32
    lib.infiniopCreateExpDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetExpWorkspaceSize.restype = c_int32
    lib.infiniopGetExpWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]


    lib.infiniopExp.restype = c_int32
    lib.infiniopExp.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyExpDescriptor.restype = c_int32
    lib.infiniopDestroyExpDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def sin_(lib):
    lib.infiniopCreateSinDescriptor.restype = c_int32
    lib.infiniopCreateSinDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSinWorkspaceSize.restype = c_int32
    lib.infiniopGetSinWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
      
    lib.infiniopSin.restype = c_int32
    lib.infiniopSin.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
      
    lib.infiniopDestroySinDescriptor.restype = c_int32
    lib.infiniopDestroySinDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]
 
@OpRegister.operator     
def tanh_(lib):
    lib.infiniopCreateTanhDescriptor.restype = c_int32
    lib.infiniopCreateTanhDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
      
    lib.infiniopGetTanhWorkspaceSize.restype = c_int32
    lib.infiniopGetTanhWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
      
    lib.infiniopTanh.restype = c_int32
    lib.infiniopTanh.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
      
    lib.infiniopDestroyTanhDescriptor.restype = c_int32
    lib.infiniopDestroyTanhDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]
    
      
@OpRegister.operator
def max_pool_(lib):
    lib.infiniopCreateMaxPoolDescriptor.restype = c_int32
    lib.infiniopCreateMaxPoolDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # output_desc
        infiniopTensorDescriptor_t,  # input_desc
        c_void_p,  # kernel_size
        c_void_p,  # strides
        c_void_p,  # pads
        c_bool,  # ceil_mode
    ]
      
    lib.infiniopGetMaxPoolWorkspaceSize.restype = c_int32
    lib.infiniopGetMaxPoolWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopMaxPool.restype = c_int32
    lib.infiniopMaxPool.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # output
        c_void_p,  # input
        c_void_p,  # stream
    ]
    
    lib.infiniopDestroyMaxPoolDescriptor.restype = c_int32
    lib.infiniopDestroyMaxPoolDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def interpolate_nearest_(lib):
    lib.infiniopCreateInterpolateNearestDescriptor.restype = c_int32
    lib.infiniopCreateInterpolateNearestDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # output_desc
        infiniopTensorDescriptor_t,  # input_desc
    ]

    lib.infiniopGetInterpolateNearestWorkspaceSize.restype = c_int32
    lib.infiniopGetInterpolateNearestWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopInterpolateNearest.restype = c_int32
    lib.infiniopInterpolateNearest.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # output
        c_void_p,  # input
        c_void_p,  # stream
    ]
      
    lib.infiniopDestroyInterpolateNearestDescriptor.restype = c_int32
    lib.infiniopDestroyInterpolateNearestDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def hardswish_(lib):
    lib.infiniopCreateHardswishDescriptor.restype = c_int32
    lib.infiniopCreateHardswishDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
      
    lib.infiniopGetHardswishWorkspaceSize.restype = c_int32
    lib.infiniopGetHardswishWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
      
    lib.infiniopHardswish.restype = c_int32
    lib.infiniopHardswish.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    
    lib.infiniopDestroyHardswishDescriptor.restype = c_int32
    lib.infiniopDestroyHardswishDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]
    

@OpRegister.operator
def where_(lib):
    lib.infiniopCreateWhereDescriptor.restype = c_int32
    lib.infiniopCreateWhereDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    
    lib.infiniopGetWhereWorkspaceSize.restype = c_int32
    lib.infiniopGetWhereWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopWhere.restype = c_int32
    lib.infiniopWhere.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyWhereDescriptor.restype = c_int32
    lib.infiniopDestroyWhereDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def reduce_max_(lib):
    lib.infiniopCreateReduceMaxDescriptor.restype = c_int32
    lib.infiniopCreateReduceMaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_size_t,
    ]

    lib.infiniopGetReduceMaxWorkspaceSize.restype = c_int32
    lib.infiniopGetReduceMaxWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopReduceMax.restype = c_int32
    lib.infiniopReduceMax.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyReduceMaxDescriptor.restype = c_int32
    lib.infiniopDestroyReduceMaxDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def batch_norm_backward_(lib):
    lib.infiniopCreateBatchNormBackwardDescriptor.restype = c_int32
    lib.infiniopCreateBatchNormBackwardDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetBatchNormBackwardWorkspaceSize.restype = c_int32
    lib.infiniopGetBatchNormBackwardWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]    
    lib.infiniopBatchNormBackward.restype = c_int32
    lib.infiniopBatchNormBackward.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyBatchNormBackwardDescriptor.restype = c_int32
    lib.infiniopDestroyBatchNormBackwardDescriptor.argtypes = [infiniopOperatorDescriptor_t]
