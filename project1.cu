#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>


// float

void cpu_matrixMultiplication_float(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void gpu_simple_matrixMultiplication_float(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void gpu_tiled_matrixMultiplication_float(const float *A, const float *B, float *C, int N) {
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;

    for (int m = 0; m < (N + 16 - 1) / 16; m++) {
        if (row < N && m * 16 + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + m * 16 + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && m * 16 + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(m * 16 + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void check_result_float(const float *cpu_C, const float *gpu_C, int N) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(cpu_C[i] - gpu_C[i]) > 1e-4) {
            printf("Wrong answer at index %d\n", i);
            return;
        }
    }
}

void run_float(int N) {
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    dim3 threads(16, 16);
    dim3 blocks((N + 16 - 1) / 16, (N + 16 - 1) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float h2d_time, kernel_time, d2h_time, total_time;

    printf("\nFloat:\n");
    
    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_matrixMultiplication_float(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;
    printf("CPU: Total: %.3f ms\n", cpu_duration.count());

    // H2D Transfer
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_time, start, stop);

    // GPU Simple
    cudaEventRecord(start);
    gpu_simple_matrixMultiplication_float<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    total_time = h2d_time + kernel_time + d2h_time;
    printf("GPU Simple: H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", h2d_time, kernel_time, d2h_time, total_time);
    check_result_float(h_C_cpu, h_C_gpu, N);

    // GPU Tiled
    cudaEventRecord(start);
    gpu_tiled_matrixMultiplication_float<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    total_time = h2d_time + kernel_time + d2h_time;
    printf("GPU Tiled : H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", h2d_time, kernel_time, d2h_time, total_time);
    check_result_float(h_C_cpu, h_C_gpu, N);

    //cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    total_time = h2d_time + kernel_time + d2h_time;
    printf("cuBLAS    : H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", h2d_time, kernel_time, d2h_time, total_time);
    check_result_float(h_C_cpu, h_C_gpu, N);
    
    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// double

void cpu_matrixMultiplication_double(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void gpu_simple_matrixMultiplication_double(const double *A, const double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void gpu_tiled_matrixMultiplication_double(const double *A, const double *B, double *C, int N) {
    __shared__ double tile_A[16][16];
    __shared__ double tile_B[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    double sum = 0.0;

    for (int m = 0; m < (N + 16 - 1) / 16; m++) {
        if (row < N && m * 16 + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + m * 16 + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && m * 16 + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(m * 16 + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void check_result_double(const double *cpu_C, const double *gpu_C, int N) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(cpu_C[i] - gpu_C[i]) > 1e-6) {
            printf("Wrong answer in double at index %d\n", i);
            return;
        }
    }
}

void run_double(int N) {
    size_t size = N * N * sizeof(double);
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C_cpu = (double *)malloc(size);
    double *h_C_gpu = (double *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (double)(rand() % 100) / 10.0;
        h_B[i] = (double)(rand() % 100) / 10.0;
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    dim3 threads(16, 16);
    dim3 blocks((N + 16 - 1) / 16, (N + 16 - 1) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float h2d_time, kernel_time, d2h_time, total_time;

    printf("\nDouble:\n");
    
    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_matrixMultiplication_double(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;
    printf("CPU: Total: %.3f ms\n", cpu_duration.count());

    // H2D
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_time, start, stop);

    // GPU Simple
    cudaEventRecord(start);
    gpu_simple_matrixMultiplication_double<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    total_time = h2d_time + kernel_time + d2h_time;
    printf("GPU Simple: H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", h2d_time, kernel_time, d2h_time, total_time);
    check_result_double(h_C_cpu, h_C_gpu, N);

    // GPU Tiled
    cudaEventRecord(start);
    gpu_tiled_matrixMultiplication_double<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    total_time = h2d_time + kernel_time + d2h_time;
    printf("GPU Tiled : H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", h2d_time, kernel_time, d2h_time, total_time);
    check_result_double(h_C_cpu, h_C_gpu, N);

    // cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0, beta = 0.0;
    cudaEventRecord(start);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    total_time = h2d_time + kernel_time + d2h_time;
    printf("cuBLAS    : H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", h2d_time, kernel_time, d2h_time, total_time);
    check_result_double(h_C_cpu, h_C_gpu, N);
    
    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// int

void cpu_matrixMultiplication_int(const int *A, const int *B, int *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void gpu_simple_matrixMultiplication_int(const int *A, const int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void gpu_tiled_matrixMultiplication_int(const int *A, const int *B, int *C, int N) {
    __shared__ int tile_A[16][16];
    __shared__ int tile_B[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    int sum = 0;

    for (int m = 0; m < (N + 16 - 1) / 16; m++) {
        if (row < N && m * 16 + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + m * 16 + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        if (col < N && m * 16 + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(m * 16 + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void run_int8_cublas(int N, const int *h_A_int, const int *h_B_int) {
    int8_t *h_A_int8 = (int8_t *)malloc(N * N * sizeof(int8_t));
    int8_t *h_B_int8 = (int8_t *)malloc(N * N * sizeof(int8_t));
    for (int i = 0; i < N * N; i++) {
        h_A_int8[i] = (int8_t)h_A_int[i];
        h_B_int8[i] = (int8_t)h_B_int[i];
    }

    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc((void **)&d_A, N * N * sizeof(int8_t));
    cudaMalloc((void **)&d_B, N * N * sizeof(int8_t));
    cudaMalloc((void **)&d_C, N * N * sizeof(int32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // H2D
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A_int8, N * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_int8, N * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_h2d;
    cudaEventElapsedTime(&ms_h2d, start, stop);

    cublasHandle_t handle;
    cublasCreate(&handle);
    int32_t alpha = 1;
    int32_t beta = 0;

    cudaEventRecord(start);
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, CUDA_R_8I, N,
        d_A, CUDA_R_8I, N,
        &beta,
        d_C, CUDA_R_32I, N,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_kernel;
    cudaEventElapsedTime(&ms_kernel, start, stop);

    // D2H
    int32_t *h_C = (int32_t *)malloc(N * N * sizeof(int32_t));
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, N * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_d2h;
    cudaEventElapsedTime(&ms_d2h, start, stop);

    float ms_total = ms_h2d + ms_kernel + ms_d2h;
    printf("cuBLAS    : H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", 
           ms_h2d, ms_kernel, ms_d2h, ms_total);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A_int8); free(h_B_int8); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}


void check_result_int(const int *cpu_C, const int *gpu_C, int N) {
    for (int i = 0; i < N * N; i++) {
        if (cpu_C[i] != gpu_C[i]) {
            printf("Wrong answer in int at index %d: CPU: %d, GPU: %d\n", i, cpu_C[i], gpu_C[i]);
            return;
        }
    }
}

void run_int(int N) {
    size_t size = N * N * sizeof(int);
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C_cpu = (int *)malloc(size);
    int *h_C_gpu = (int *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    dim3 threads(16, 16);
    dim3 blocks((N + 16 - 1) / 16, (N + 16 - 1) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float h2d_time, kernel_time, d2h_time, total_time;

    printf("\nInt:\n");

    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_matrixMultiplication_int(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;
    printf("CPU: Total: %.3f ms\n", cpu_duration.count());

    // H2D
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_time, start, stop);

    // GPU Simple
    cudaEventRecord(start);
    gpu_simple_matrixMultiplication_int<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    total_time = h2d_time + kernel_time + d2h_time;
    printf("GPU Simple: H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", h2d_time, kernel_time, d2h_time, total_time);
    check_result_int(h_C_cpu, h_C_gpu, N);

    // GPU Tiled
    cudaEventRecord(start);
    gpu_tiled_matrixMultiplication_int<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    total_time = h2d_time + kernel_time + d2h_time;
    printf("GPU Tiled : H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n", h2d_time, kernel_time, d2h_time, total_time);
    check_result_int(h_C_cpu, h_C_gpu, N);

    // cuBLAS
    run_int8_cublas(N, h_A, h_B);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    int N;

    printf("N = ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("Invalid input\n");
        return 1;
    }

    run_float(N);
    run_double(N);
    run_int(N);

    return 0;
}
