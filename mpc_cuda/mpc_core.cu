#include <iostream>
#include "../mpc_keys/keys_type.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
__global__ void matrixMulKernel(const int64_t* A, const int64_t* B, int64_t* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引

    if (row < rowsA && col < colsB) {
        int64_t sum = 0;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col]; // 确保索引正确
        }
        C[row * colsB + col] = sum; // 结果存储
    }
}
// 简单的CUDA核函数
__global__ void warmupKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;  // 简单的操作，增加每个元素的值
    }
}

__global__ void MatrixLinearKernel(const int64_t a, const int64_t* A, const int64_t b, const int64_t* B, const int64_t c,int64_t* C, int rowsA, int colsA) {

    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引

    if (row < rowsA && col < colsA) {
        C[row * colsA + col] = a * A[row * colsA + col] + b * B[row * colsA + col] + c;
    }

}

// Function to allocate device memory and transfer MM_Keys data to GPU
bool MMKeysToGPU(const MM_Keys& keys, int64_t** d_R_A, int64_t** d_R_B, int64_t** d_R_AB) {
    size_t sizeA = keys.R_A.size() * sizeof(int64_t);
    size_t sizeB = keys.R_B.size() * sizeof(int64_t);
    size_t sizeC = keys.R_AB.size() * sizeof(int64_t);

    // Allocate device memory
    if (cudaMalloc(d_R_A, sizeA) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for R_A" << std::endl;
        return false;
    }
    if (cudaMalloc(d_R_B, sizeB) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for R_B" << std::endl;
        cudaFree(*d_R_A);
        return false;
    }
    if (cudaMalloc(d_R_AB, sizeC) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for R_AB" << std::endl;
        cudaFree(*d_R_A);
        cudaFree(*d_R_B);
        return false;
    }

    // Transfer data to GPU
    cudaMemcpy(*d_R_A, keys.R_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_R_B, keys.R_B.data(), sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_R_AB, keys.R_AB.data(), sizeC, cudaMemcpyHostToDevice);

    return true;
}


extern "C" void cudaMatrixMultiply(const int64_t* A, const int64_t* B, int64_t* C, int rowsA, int colsA, int colsB) {

    // 获取当前使用的设备
    int currentDevice;
    cudaGetDevice(&currentDevice);
    std::cout << "Currently using CUDA device: " << currentDevice << std::endl;

    // 矩阵大小
    size_t sizeA = rowsA * colsA * sizeof(int64_t);
    size_t sizeB = colsA * colsB * sizeof(int64_t);
    size_t sizeC = rowsA * colsB * sizeof(int64_t);

    // 分配设备内存
    int64_t *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // 拷贝数据到设备
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // 定义 CUDA 的线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + 15) / 16, (rowsA + 15) / 16);

    // 启动 CUDA 核函数
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // 同步，检查错误
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    }


extern "C" void cudaMatrixLinear(const int64_t a, const int64_t* A, const int64_t b, const int64_t* B, const int64_t c,int64_t* C, int rowsA, int colsA) {
    // 获取当前使用的设备
    int currentDevice;
    cudaGetDevice(&currentDevice);
    std::cout << "Currently using CUDA device: " << currentDevice << std::endl;

    // 矩阵大小
    size_t sizeA = rowsA * colsA * sizeof(int64_t);

    // 分配设备内存
    int64_t *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeA);
    cudaMalloc((void**)&d_C, sizeA);

    // 拷贝数据到设备
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeA, cudaMemcpyHostToDevice);

    // 定义 CUDA 的线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsA + 15) / 16, (rowsA + 15) / 16);

    // 启动 CUDA 核函数
    MatrixLinearKernel<<<blocksPerGrid, threadsPerBlock>>>(a, d_A, b, d_B, c, d_C, rowsA, colsA);

    // 同步，检查错误
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(C, d_C, sizeA, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

extern "C" void cudaMPC_MM(MatrixRowMajor &Public_C, MatrixRowMajor &Public_A, MatrixRowMajor &Public_B, MM_Keys &keys, int party) {
    // 获取当前使用的设备
    int currentDevice;
    cudaGetDevice(&currentDevice);
    std::cout << "Currently using CUDA device: " << currentDevice << std::endl;

    cudaEvent_t start1, stop1, start2, stop2, start3, stop3, start4, stop4;
    cudaEventCreate(&start1);   
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);   
    cudaEventCreate(&stop2);
    cudaEventCreate(&start3);   
    cudaEventCreate(&stop3);
    cudaEventCreate(&start4);   
    cudaEventCreate(&stop4);

    
    // 矩阵大小
    int rowsA = Public_A.rows();
    int colsA = Public_A.cols();
    int colsB = Public_B.cols();
    size_t sizeA = Public_A.size() * sizeof(int64_t);
    size_t sizeB = Public_B.size() * sizeof(int64_t);
    size_t sizeC = Public_C.size() * sizeof(int64_t);

    // 分配设备内存
    int64_t *d_A, *d_B, *d_C, *d_R_A, *d_R_B, *d_R_AB, *d_temp, *d_temp2;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);
    cudaMalloc((void**)&d_temp, sizeC);
    cudaMalloc((void**)&d_temp2, sizeC);

    cudaEventRecord(start4);
    // 拷贝数据到设备
    cudaMemcpy(d_A, Public_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, Public_B.data(), sizeB, cudaMemcpyHostToDevice);


    cudaEventRecord(stop4);

    cudaEventRecord(start2);
    // Allocate and transfer data to GPU
    if (!MMKeysToGPU(keys, &d_R_A, &d_R_B, &d_R_AB)) {
        std::cerr << "Failed to allocate and transfer data to GPU" << std::endl;
        return;
    }
    cudaEventRecord(stop2);


    // 定义 CUDA 的线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsA + 15) / 16, (rowsA + 15) / 16);

    
    // 启动 CUDA 核函数
    cudaEventRecord(start1);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_temp, rowsA, colsA, colsB);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_R_B, d_temp2, rowsA, colsA, colsB);
    MatrixLinearKernel<<<blocksPerGrid, threadsPerBlock>>>((party-1), d_temp, -1, d_temp2, 0, d_temp, rowsA, colsB);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_R_A, d_B, d_temp2, rowsA, colsA, colsB);
    MatrixLinearKernel<<<blocksPerGrid, threadsPerBlock>>>(1, d_temp, -1, d_temp2, 0, d_temp, rowsA, colsB);
    MatrixLinearKernel<<<blocksPerGrid, threadsPerBlock>>>(1, d_temp, 1, d_R_AB, 0, d_C, rowsA, colsB);
    // 同步，检查错误
    cudaDeviceSynchronize();
    cudaEventRecord(stop1);


    // 拷贝结果回主机
    cudaEventRecord(start3);
    cudaMemcpy(Public_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop3);

    float time1 = 0, time2 = 0, time3 = 0, time4 = 0;
    cudaEventElapsedTime(&time1, start1, stop1);
    cudaEventElapsedTime(&time2, start2, stop2);
    cudaEventElapsedTime(&time3, start3, stop3);
    cudaEventElapsedTime(&time4, start4, stop4);
    // 打印结果
    printf("matrixMulKernel time: %.3f ms\n", time1);
    printf("MMKeysToGPU time: %.3f ms\n", time2);
    printf("cudaMemcpyDeviceToHost time: %.3f ms\n", time3);
    printf("alloc memory time: %.3f ms\n", time4);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_R_A);
    cudaFree(d_R_B);
    cudaFree(d_R_AB);   
    cudaFree(d_temp);
    cudaFree(d_temp2);
}

// CUDA warmup函数 
extern "C" void cudaWarmup(int size, int party) {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    int deviceToUse = party; // 指定要使用的设备编号
    if (deviceToUse < deviceCount) {
        cudaSetDevice(deviceToUse);
        std::cout << "Using CUDA device: " << deviceToUse << std::endl;
    } else {
        std::cerr << "Invalid device number." << std::endl;
        return ;
    }

    int* d_data;
    size_t bytes = size * sizeof(int);
    // 分配设备内存
    cudaMalloc(&d_data, bytes);

    // 定义CUDA核函数的网格和块大小
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // 启动CUDA核函数
    warmupKernel<<<gridSize, blockSize>>>(d_data, size);

    // 同步设备，确保核函数执行完成
    cudaDeviceSynchronize();

    // 释放设备内存
    cudaFree(d_data);
}