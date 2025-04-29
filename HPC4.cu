//hpc 4a

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

#define BLOCK_SIZE 256
#define CUDA_CORES 768

__global__ void vectorAddShared(int* A, int* B, int* C, int n) {
    __shared__ int s_A[BLOCK_SIZE];
    __shared__ int s_B[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        s_A[threadIdx.x] = A[idx];
        s_B[threadIdx.x] = B[idx];
        __syncthreads();

        C[idx] = s_A[threadIdx.x] + s_B[threadIdx.x];
    }
}

void vectorAddCPU(int* A, int* B, int* C, int n) {
    for (int i = 0; i < n; ++i)
        C[i] = A[i] + B[i];
}

int main() {
    int sizes[5] = {100000, 5000000, 8000000, 10000000, 30000000};
    printf("\nName: Pratyush Funde  Class: BE A  Roll NO :41013)\n");
    printf("\nVector Addition Benchmark (Shared Memory)\n");
    printf("---------------------------------------------------------------\n");
    printf("| Vector Size | CPU Time(s) | GPU Time(s) | Speedup | Efficiency |\n");
    printf("---------------------------------------------------------------\n");

    for (int i = 0; i < 5; i++) {
        int N = sizes[i];
        int *h_A = (int*)malloc(N * sizeof(int));
        int *h_B = (int*)malloc(N * sizeof(int));
        int *h_C_CPU = (int*)malloc(N * sizeof(int));
        int *h_C_GPU = (int*)malloc(N * sizeof(int));

        for (int j = 0; j < N; ++j) {
            h_A[j] = rand() % 100;
            h_B[j] = rand() % 100;
        }

        // CPU time
        auto start_cpu = std::chrono::high_resolution_clock::now();
        vectorAddCPU(h_A, h_B, h_C_CPU, N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
        double cpu_time = cpu_duration.count();

        // Allocate device memory
        int *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, N * sizeof(int));
        cudaMalloc((void**)&d_B, N * sizeof(int));
        cudaMalloc((void**)&d_C, N * sizeof(int));

        cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

        // GPU time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cudaEventRecord(start);
        vectorAddShared<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms, start, stop);
        double gpu_time = gpu_time_ms / 1000.0;

        cudaMemcpy(h_C_GPU, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

        double speedup = cpu_time / gpu_time;
        double efficiency = speedup / CUDA_CORES;

        printf("| %11d | %11.6f | %11.6f | %7.2f | %9.6f |\n",
               N, cpu_time, gpu_time, speedup, efficiency);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("---------------------------------------------------------------\n");
    return 0;
}


//hpc 4 b>

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>

#define CUDA_CORES 768  // GTX 1050Ti CUDA cores

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

/********************** CUDA Kernel: Matrix Multiplication **********************/
__global__ void matrixMulCanon(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/********************** Sequential Matrix Multiplication **********************/
void sequentialMatrixMul(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/********************** Run Matrix Multiplication Test **********************/
void runMatrixMultiplication(int N) {
    std::vector<int> A(N * N), B(N * N), C(N * N);

    for (int i = 0; i < N * N; ++i) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(int)));

    cudaMemcpy(d_A, A.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    // CPU execution time measurement
    auto startSeq = std::chrono::high_resolution_clock::now();
    sequentialMatrixMul(A, B, C, N);
    auto endSeq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seqTime = endSeq - startSeq;

    // GPU execution time measurement
    float gpuTime;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);

    cudaEventRecord(start);
    matrixMulCanon<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpuTime, start, end);

    // Check for CUDA errors after kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    double speedup = seqTime.count() / (gpuTime / 1000.0);
    double efficiency = speedup / CUDA_CORES;

    std::cout << std::setw(15) << N 
              << std::setw(20) << seqTime.count() 
              << std::setw(20) << gpuTime 
              << std::setw(20) << speedup 
              << std::setw(20) << efficiency 
              << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/********************** Main Function **********************/
int main() {
std::cout << "\n";
   std::cout << "\n";
 std::cout << "Name: Pratyush Funde Roll no: 41013 CLass: BE A\n";
 std::cout << "\n";
 std::cout << "\n";
    std::cout << "\n==== Matrix Multiplication Tests ====\n";
    std::cout << std::setw(15) << "Matrix Size" 
              << std::setw(20) << "CPU Time (s)" 
              << std::setw(20) << "GPU Time (ms)" 
              << std::setw(20) << "Speedup" 
              << std::setw(20) << "Efficiency" 
              << std::endl;
    std::cout << std::string(95, '-') << "\n";

    int matrixTestCases[] = {128,256,768,1500,2000,3000};  // Large matrix sizes
    for (int i = 0; i < 5; i++) {
        runMatrixMultiplication(matrixTestCases[i]);
    }

    return 0;
}

