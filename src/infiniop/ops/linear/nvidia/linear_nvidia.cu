#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "linear_nvidia.cuh"

namespace op::linear::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;

    cublasLtMatmulDesc_t lt_desc;
    cublasLtMatrixLayout_t a_layout, b_layout, c_layout, d_layout;
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
        , INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = op::gemm::MatmulInfo::create(c_desc, a_desc, b_desc, op::gemm::MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    cublasLtMatmulDesc_t lt_desc = NULL;
    cublasLtMatrixLayout_t a_layout = NULL, b_layout = NULL, c_layout = NULL, d_layout = NULL;

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal(), lt_desc, a_layout, b_layout, c_layout, d_layout},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
            float alpha,                                         
            const void *a,                                       
            const void *b,                                       
            float beta,                                          
            const void *c,                                       
            void *d,                                             
            void *workspace, size_t workspace_size,              
            void *stream) const {

    cudaDataType a_type, b_type, c_type, d_type, scale_type;
    // CUBLASLT_MATMUL_DESC_SCALE_TYPE scale_type;
    cublasComputeType_t compute_type;
    // cudaDataType_t 
    a_type = b_type = c_type = d_type = CUDA_R_32F;
    compute_type = CUBLAS_COMPUTE_32F;
    scale_type = CUDA_R_32F;

    // cublasLtMatmulDesc_t lt_desc;
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&_opaque->lt_desc, compute_type, scale_type));
    
    if (_info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));

    // cublasLtMatrixLayout_t a_layout, b_layout, c_layout, d_layout;

    if (op_a)  {
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&_opaque->a_layout, a_type, _info.k, _info.m, _info.a_matrix.ld()));
    } else {
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&_opaque->a_layout, a_type, _info.m, _info.k, _info.a_matrix.ld()));
    }

    if (op_b) {
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&_opaque->b_layout, b_type, _info.n, _info.k, _info.b_matrix.ld()));
    } else {
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&_opaque->b_layout, b_type, _info.k, _info.n, _info.b_matrix.ld()));
    }

    //cuBlasLt requires C in fp8 mode to be BF16 or FP32
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&_opaque->c_layout, c_type, _info.m, _info.n, _info.c_matrix.ld()));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&_opaque->d_layout, d_type, _info.m, _info.n, _info.c_matrix.ld()));

    int batch = static_cast<int>(_info.batch);

    if (batch) {
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(_opaque->a_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(_opaque->b_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(_opaque->c_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(_opaque->d_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));

        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(_opaque->a_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &_info.a_matrix.stride, sizeof(_info.a_matrix.stride)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(_opaque->b_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &_info.b_matrix.stride, sizeof(_info.b_matrix.stride)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(_opaque->c_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &_info.c_matrix.stride, sizeof(_info.c_matrix.stride)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(_opaque->d_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &_info.c_matrix.stride, sizeof(_info.c_matrix.stride)));
    }

    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(_opaque->lt_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    CHECK_STATUS(_opaque->internal->useCublasLt(
        (cudaStream_t)stream,
        [&](cublasLtHandle_t handle) {
            CHECK_CUBLASLT(
                cublasLtMatmul(handle, 
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
