#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <limits>
#include <cstdlib>

#define CUDA_CORES 768
#define BLOCK_SIZE 256
#define WARP_SIZE 32

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduceSum(int* input, unsigned long long* output, int n) {
    __shared__ int shared[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < n) ? input[tid] : 0;

    val = warpReduceSum(val);

    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
        val = warpReduceSum(val);
    }

    if (threadIdx.x == 0) atomicAdd(output, (unsigned long long)val);
}

long long sequentialSum(const std::vector<int>& data) {
    long long sum = 0;
    for (int val : data) sum += val;
    return sum;
}

int main() {
    std::vector<long long> sizes = {20000000, 30000000, 40000000, 50000000};
    std::vector<int> maxValues = {100, 500, 1000, 1500};

    std::cout << "\n";
    std::cout << "Name: Pratyush Funde Roll no: 41013 Class: BE A\n";
    std::cout << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "| Input Size | Max Value | CPU Sum     | GPU Sum     | CPU Time (s) | GPU Time (s) | Speedup | Efficiency |\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------------\n";

    for (size_t i = 0; i < sizes.size(); i++) {
        long long n = sizes[i];
        int maxVal = maxValues[i];
        std::vector<int> data(n);

        for (long long j = 0; j < n; ++j)
            data[j] = rand() % maxVal;

        int* d_input;
        unsigned long long* d_sum;
        unsigned long long h_sum = 0;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));

        CUDA_CHECK(cudaMemcpy(d_input, data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));

        int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        auto cpuStart = std::chrono::high_resolution_clock::now();
        long long cpuSum = sequentialSum(data);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpuTime = cpuEnd - cpuStart;

        cudaEvent_t start, stop;
        float gpuTimeMs = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        reduceSum<<<numBlocks, BLOCK_SIZE>>>(d_input, d_sum, n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTimeMs, start, stop);

        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        double speedup = cpuTime.count() / (gpuTimeMs / 1000.0);
        double efficiency = speedup / CUDA_CORES;

        std::cout << "| " << std::setw(10) << n
                  << " | " << std::setw(9) << maxVal
                  << " | " << std::setw(11) << cpuSum
                  << " | " << std::setw(11) << h_sum
                  << " | " << std::setw(12) << std::fixed << std::setprecision(6) << cpuTime.count()
                  << " | " << std::setw(12) << gpuTimeMs / 1000.0
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << speedup
                  << " | " << std::setw(10) << std::fixed << std::setprecision(6) << efficiency
                  << " |\n";

        cudaFree(d_input);
        cudaFree(d_sum);
    }

    std::cout << "--------------------------------------------------------------------------------------------------------------------\n";
    return 0;
}

