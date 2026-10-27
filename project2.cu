#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// float

void run_cusparse_float(int N, int nnz, int *hA_RowOffsets, int *hA_ColIndices) {
    float *hA_Values = (float *)malloc(nnz * sizeof(float));
    float *hB = (float *)malloc(N * N * sizeof(float));
    float *hC = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < nnz; i++) hA_Values[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N * N; i++) {
        hB[i] = (float)rand() / RAND_MAX;
        hC[i] = 0.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int *dA_RowOffsets, *dA_ColIndices;
    float *dA_Values, *dB, *dC;
    cudaMalloc((void**)&dA_RowOffsets, (N + 1) * sizeof(int));
    cudaMalloc((void**)&dA_ColIndices, nnz * sizeof(int));
    cudaMalloc((void**)&dA_Values, nnz * sizeof(float));
    cudaMalloc((void**)&dB, N * N * sizeof(float));
    cudaMalloc((void**)&dC, N * N * sizeof(float));

    cudaEventRecord(start);
    cudaMemcpy(dA_RowOffsets, hA_RowOffsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_ColIndices, hA_ColIndices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_Values, hA_Values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_h2d; cudaEventElapsedTime(&ms_h2d, start, stop);

    cusparseHandle_t handle; cusparseCreate(&handle);
    cusparseSpMatDescr_t matA; cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL; size_t bufferSize = 0;

    cusparseCreateCsr(&matA, N, N, nnz, dA_RowOffsets, dA_ColIndices, dA_Values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, N, N, N, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, N, N, N, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cudaEventRecord(start);
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_kernel; cudaEventElapsedTime(&ms_kernel, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(hC, dC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_d2h; cudaEventElapsedTime(&ms_d2h, start, stop);

    printf("Float     : H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", 
           ms_h2d, ms_kernel, ms_d2h, ms_h2d + ms_kernel + ms_d2h);

    cusparseDestroySpMat(matA); cusparseDestroyDnMat(matB); cusparseDestroyDnMat(matC); cusparseDestroy(handle);
    cudaFree(dA_RowOffsets); cudaFree(dA_ColIndices); cudaFree(dA_Values); cudaFree(dB); cudaFree(dC); cudaFree(dBuffer);
    free(hA_Values); free(hB); free(hC); cudaEventDestroy(start); cudaEventDestroy(stop);
}

// double

void run_cusparse_double(int N, int nnz, int *hA_RowOffsets, int *hA_ColIndices) {
    double *hA_Values = (double *)malloc(nnz * sizeof(double));
    double *hB = (double *)malloc(N * N * sizeof(double));
    double *hC = (double *)malloc(N * N * sizeof(double));

    for (int i = 0; i < nnz; i++) hA_Values[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < N * N; i++) {
        hB[i] = (double)rand() / RAND_MAX;
        hC[i] = 0.0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int *dA_RowOffsets, *dA_ColIndices;
    double *dA_Values, *dB, *dC;
    cudaMalloc((void**)&dA_RowOffsets, (N + 1) * sizeof(int));
    cudaMalloc((void**)&dA_ColIndices, nnz * sizeof(int));
    cudaMalloc((void**)&dA_Values, nnz * sizeof(double));
    cudaMalloc((void**)&dB, N * N * sizeof(double));
    cudaMalloc((void**)&dC, N * N * sizeof(double));

    cudaEventRecord(start);
    cudaMemcpy(dA_RowOffsets, hA_RowOffsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_ColIndices, hA_ColIndices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_Values, hA_Values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_h2d; cudaEventElapsedTime(&ms_h2d, start, stop);

    cusparseHandle_t handle; cusparseCreate(&handle);
    cusparseSpMatDescr_t matA; cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL; size_t bufferSize = 0;

    cusparseCreateCsr(&matA, N, N, nnz, dA_RowOffsets, dA_ColIndices, dA_Values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnMat(&matB, N, N, N, dB, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, N, N, N, dC, CUDA_R_64F, CUSPARSE_ORDER_ROW);

    double alpha = 1.0, beta = 0.0;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cudaEventRecord(start);
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_kernel; cudaEventElapsedTime(&ms_kernel, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(hC, dC, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_d2h; cudaEventElapsedTime(&ms_d2h, start, stop);

    printf("Double    : H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", 
           ms_h2d, ms_kernel, ms_d2h, ms_h2d + ms_kernel + ms_d2h);

    cusparseDestroySpMat(matA); cusparseDestroyDnMat(matB); cusparseDestroyDnMat(matC); cusparseDestroy(handle);
    cudaFree(dA_RowOffsets); cudaFree(dA_ColIndices); cudaFree(dA_Values); cudaFree(dB); cudaFree(dC); cudaFree(dBuffer);
    free(hA_Values); free(hB); free(hC); cudaEventDestroy(start); cudaEventDestroy(stop);
}

// int

void run_cusparse_int(int N, int nnz, int *hA_RowOffsets, int *hA_ColIndices) {
    int8_t *hA_Values = (int8_t *)malloc(nnz * sizeof(int8_t));
    int8_t *hB = (int8_t *)malloc(N * N * sizeof(int8_t));
    int32_t *hC = (int32_t *)malloc(N * N * sizeof(int32_t));

    for (int i = 0; i < nnz; i++) hA_Values[i] = (int8_t)(rand() % 10);
    for (int i = 0; i < N * N; i++) {
        hB[i] = (int8_t)(rand() % 10);
        hC[i] = 0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int *dA_RowOffsets, *dA_ColIndices;
    int8_t *dA_Values, *dB;
    int32_t *dC;
    cudaMalloc((void**)&dA_RowOffsets, (N + 1) * sizeof(int));
    cudaMalloc((void**)&dA_ColIndices, nnz * sizeof(int));
    cudaMalloc((void**)&dA_Values, nnz * sizeof(int8_t));
    cudaMalloc((void**)&dB, N * N * sizeof(int8_t));
    cudaMalloc((void**)&dC, N * N * sizeof(int32_t));

    cudaEventRecord(start);
    cudaMemcpy(dA_RowOffsets, hA_RowOffsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_ColIndices, hA_ColIndices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_Values, hA_Values, nnz * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_h2d; cudaEventElapsedTime(&ms_h2d, start, stop);

    cusparseHandle_t handle; cusparseCreate(&handle);
    cusparseSpMatDescr_t matA; cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL; size_t bufferSize = 0;

    cusparseCreateCsr(&matA, N, N, nnz, dA_RowOffsets, dA_ColIndices, dA_Values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I);
    cusparseCreateDnMat(&matB, N, N, N, dB, CUDA_R_8I, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, N, N, N, dC, CUDA_R_32I, CUSPARSE_ORDER_ROW);

    int32_t alpha = 1, beta = 0;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32I, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cudaEventRecord(start);
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC, CUDA_R_32I, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_kernel; cudaEventElapsedTime(&ms_kernel, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(hC, dC, N * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_d2h; cudaEventElapsedTime(&ms_d2h, start, stop);

    printf("Int8      : H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", 
           ms_h2d, ms_kernel, ms_d2h, ms_h2d + ms_kernel + ms_d2h);

    cusparseDestroySpMat(matA); cusparseDestroyDnMat(matB); cusparseDestroyDnMat(matC); cusparseDestroy(handle);
    cudaFree(dA_RowOffsets); cudaFree(dA_ColIndices); cudaFree(dA_Values); cudaFree(dB); cudaFree(dC); cudaFree(dBuffer);
    free(hA_Values); free(hB); free(hC); cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    srand((unsigned)time(NULL));

    int N;
    printf("N = ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("Invalid input\n");
        return 1;
    }

    int *hA_RowOffsets = (int *)malloc((N + 1) * sizeof(int));
    int *hA_ColIndices = (int *)malloc(N * N * sizeof(int));
    int nnz = 0;
    
    for (int i = 0; i < N; i++) {
        hA_RowOffsets[i] = nnz;
        for (int j = 0; j < N; j++) {
            if ((float)rand() / RAND_MAX > 0.9f) {
                hA_ColIndices[nnz] = j;
                nnz++;
            }
        }
    }
    hA_RowOffsets[N] = nnz;

    printf("\ncuSPARSE (A: 90%% Zeros, NNZ = %d):\n", nnz);

    run_cusparse_float(N, nnz, hA_RowOffsets, hA_ColIndices);
    run_cusparse_double(N, nnz, hA_RowOffsets, hA_ColIndices);
    run_cusparse_int(N, nnz, hA_RowOffsets, hA_ColIndices);

    free(hA_RowOffsets);
    free(hA_ColIndices);

    return 0;
}
