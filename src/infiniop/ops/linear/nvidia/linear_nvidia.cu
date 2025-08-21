#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "linear_nvidia.cuh"

namespace op::linear::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;

    cublasLtMatmulDesc_t lt_desc;
    cublasLtMatrixLayout_t a_layout, b_layout, c_layout, d_layout;
    cudaDataType bias_type;
};

Descriptor::~Descriptor() {
    cublasLtMatmulDescDestroy(_opaque->lt_desc);
    cublasLtMatrixLayoutDestroy(_opaque->a_layout);
    cublasLtMatrixLayoutDestroy(_opaque->b_layout);
    cublasLtMatrixLayoutDestroy(_opaque->c_layout);
    cublasLtMatrixLayoutDestroy(_opaque->d_layout);
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t d_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t c_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = d_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F8_E4M3, INFINI_DTYPE_F8_E5M2
        , INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

    auto result = op::gemm::MatmulInfo::create(c_desc, a_desc, b_desc, op::gemm::MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    cublasLtMatmulDesc_t lt_desc = NULL;
    cublasLtMatrixLayout_t a_layout = NULL, b_layout = NULL, c_layout = NULL, d_layout = NULL;

    cudaDataType a_type, b_type, c_type, d_type, scale_type, bias_type;
    cublasComputeType_t compute_type;

    switch (a_desc->dtype()) {
        case INFINI_DTYPE_F8_E4M3:
            a_type = CUDA_R_8F_E4M3;
            compute_type = CUBLAS_COMPUTE_32F;
            scale_type = CUDA_R_32F;
            switch (b_desc->dtype()) {
                case INFINI_DTYPE_F8_E4M3:
                    b_type = CUDA_R_8F_E4M3;
                    switch (c_desc->dtype()) {
                        case INFINI_DTYPE_BF16:
                            c_type = CUDA_R_16BF;
                            switch(d_desc->dtype()) {
                                case INFINI_DTYPE_BF16:
                                    d_type = CUDA_R_16BF;
                                    bias_type = CUDA_R_16BF;
                                    break;
                                case INFINI_DTYPE_F8_E4M3:
                                    d_type = CUDA_R_8F_E4M3;
                                    bias_type = CUDA_R_16BF;
                                    break;
                                default:
                                    return INFINI_STATUS_BAD_TENSOR_DTYPE;
                            }
                            break;
                        case INFINI_DTYPE_F16:
                            c_type = CUDA_R_16F;
                            switch(d_desc->dtype()) {
                                case INFINI_DTYPE_F16:
                                    d_type = CUDA_R_16F;
                                    bias_type = CUDA_R_16F;
                                    break;
                                case INFINI_DTYPE_F8_E4M3:
                                    d_type = CUDA_R_8F_E4M3;
                                    bias_type = CUDA_R_16F;
                                    break;
                                default:
                                    return INFINI_STATUS_BAD_TENSOR_DTYPE;
                            }
                            break;
                        case INFINI_DTYPE_F32:
                            c_type = CUDA_R_32F;
                            switch(d_desc->dtype()) {
                                case INFINI_DTYPE_F32:
                                    d_type = CUDA_R_32F;
                                    bias_type = CUDA_R_32F;
                                    break;
                                default:
                                    return INFINI_STATUS_BAD_TENSOR_DTYPE;
                            }
                            break;
                        default:
                            return INFINI_STATUS_NOT_IMPLEMENTED;
                    }
                    break;
                default:
                    return INFINI_STATUS_NOT_IMPLEMENTED;
            }
            break;

        case INFINI_DTYPE_F16:
            a_type = b_type = c_type = d_type = CUDA_R_16F;
            bias_type = CUDA_R_16F;
            compute_type = CUBLAS_COMPUTE_32F;
            scale_type = CUDA_R_32F;
            break;

        case INFINI_DTYPE_BF16:
            a_type = b_type = c_type = d_type = CUDA_R_16BF;
            bias_type = CUDA_R_16BF;
            compute_type = CUBLAS_COMPUTE_32F;
            scale_type = CUDA_R_32F;
            break;

        case INFINI_DTYPE_F32:
            a_type = b_type = c_type = d_type = CUDA_R_32F;
            bias_type = CUDA_R_16BF;
            compute_type = CUBLAS_COMPUTE_32F;
            scale_type = CUDA_R_32F;
            break;
        
        default:
            return INFINI_STATUS_NOT_IMPLEMENTED;
    }

    
    auto info = result.take();
    auto op_a = info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    std::cout << "op_a: " << (int)op_a << "op_b: " << (int)op_b <<std::endl;
    
    /* To use tensor- or block-scaled FP8 kernels:
     * A must be transposed and B non-transposed (The “TN” format) 
     * on Ada (compute capability 8.9), Hopper (compute capability 9.0), 
     * and Blackwell GeForce (compute capability 12.x) GPUs.
     */
    if (a_type == CUDA_R_8F_E4M3 || b_type == CUDA_R_8F_E4M3) {
        op_a = CUBLAS_OP_T;
        op_b = CUBLAS_OP_N;
    }

    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&lt_desc, compute_type, scale_type));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
    
    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    if (op_a)  {
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&a_layout, a_type, info.k, info.m, info.a_matrix.ld()));
    } else {
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&a_layout, a_type, info.m, info.k, info.a_matrix.ld()));
    }

    if (op_b) {
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&b_layout, b_type, info.n, info.k, info.b_matrix.ld()));
    } else {
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&b_layout, b_type, info.k, info.n, info.b_matrix.ld()));
    }

    //cuBlasLt requires C in fp8 mode to be BF16 or FP32
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&c_layout, c_type, info.m, info.n, info.c_matrix.ld()));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&d_layout, d_type, info.m, info.n, info.c_matrix.ld()));

    int batch = static_cast<int>(info.batch);

    if (batch) {
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(c_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));

        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &info.a_matrix.stride, sizeof(info.a_matrix.stride)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &info.b_matrix.stride, sizeof(info.b_matrix.stride)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(c_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &info.c_matrix.stride, sizeof(info.c_matrix.stride)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &info.c_matrix.stride, sizeof(info.c_matrix.stride)));
    }

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal(), lt_desc, a_layout, b_layout, c_layout, d_layout, bias_type},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
            float alpha,                                         
            const void *a,    
            const void *a_scale,                                   
            const void *b, 
            const void *b_scale,                                      
            float beta,                                          
            const void *c,         
            const void *c_scale,  
            const void *bias,                            
            void *d,              
            const void *d_scale,                            
            void *workspace, 
            size_t workspace_size,              
            void *stream) const {
    cublasComputeType_t compute_type;
    cudaDataType a_type, b_type;
    size_t *sizeWritten = 0;
    int returnedResults = 0;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtEpilogue_t epilogue;
    
    if (_info.is_transed) {
        std::swap(a, b);
    }
    std::cout << "is_transed" << _info.is_transed <<std::endl;
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutGetAttribute(_opaque->a_layout, CUBLASLT_MATRIX_LAYOUT_TYPE, &a_type, sizeof(a_type), sizeWritten));
    CHECK_CUBLASLT(cublasLtMatrixLayoutGetAttribute(_opaque->b_layout, CUBLASLT_MATRIX_LAYOUT_TYPE, &b_type, sizeof(b_type), sizeWritten));

    if( (a_type  == CUDA_R_8F_E4M3 || b_type == CUDA_R_8F_E4M3)  &&
        (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)c % 16) != 0 || ((uintptr_t)d % 16) != 0)) {
        return INFINI_STATUS_NOT_ALIGNED;
    }

    if(a_scale) {
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    }
    if(b_scale) {
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &b_scale, sizeof(b_scale)));
    }
    if(c_scale) {
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &c_scale, sizeof(c_scale)));
    }
    if(d_scale) {
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale, sizeof(d_scale)));
    }

    if(bias) {
        epilogue = CUBLASLT_EPILOGUE_BIAS;
        // std::cout << "bias:1111" << std::endl;
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &_opaque->bias_type, sizeof(_opaque->bias_type)));
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
   
    CHECK_STATUS(_opaque->internal->useCublasLt(
        (cudaStream_t)stream,
        [&](cublasLtHandle_t handle) {
            CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
                handle, 
                _opaque->lt_desc, 
                _opaque->a_layout, 
                _opaque->b_layout, 
                _opaque->c_layout, 
                _opaque->d_layout, 
                preference, 
                1, 
                &heuristicResult, 
                &returnedResults));
            if (returnedResults == 0) 
                return INFINI_STATUS_NOT_IMPLEMENTED;
            
            CHECK_CUBLASLT(
                cublasLtMatmul(
                    handle, 
                    _opaque->lt_desc, 
                    &alpha, 
                    a, 
                    _opaque->a_layout, 
                    b, 
                    _opaque->b_layout, 
                    &beta, 
                    c, 
                    _opaque->c_layout, 
                    d, 
                    _opaque->d_layout, 
                    nullptr, 
                    workspace, 
                    workspace_size, 
                    reinterpret_cast<cudaStream_t>(stream)));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::nvidia
