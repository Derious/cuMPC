#include <cuda_runtime.h>
#include <iostream>

int main() {
    const uint64_t SIZE = 30ULL * 1024ULL * 1024ULL; // 3000MB in bytes
    
    // 分配主机内存
    char* h_data = (char*)malloc(SIZE);
    
    // 初始化数据
    for(uint64_t i = 0; i < SIZE; i++) {
        h_data[i] = i % 256;
    }
    
    // 分配设备内存
    char* d_data;
    cudaMalloc(&d_data, SIZE);
    
    // 创建 CUDA 事件来计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 开始计时
    cudaEventRecord(start);
    
    // 传输数据到 GPU
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    
    // 结束计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算耗时（毫秒）
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 输出结果
    std::cout << "Data size: " << SIZE / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Transfer time: " << milliseconds << " ms" << std::endl;
    std::cout << "Bandwidth: " << (SIZE / milliseconds) / (1024.0 * 1024.0) << " GB/s" << std::endl;
    
    // 清理
    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
} 