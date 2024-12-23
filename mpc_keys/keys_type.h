#pragma once
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include "uint128_type.h"
using namespace Eigen;

#define MSB_KEY_SIZE 1203
#define LUT_KEY_SIZE 72
inline void getKeyBuf(uint8_t*&startPtr, uint8_t *&curPtr, size_t bufSize)
{
    // printf("Getting key buf\n");
    startPtr = (uint8_t *)malloc(bufSize);
    curPtr = startPtr;
}
inline void writeKeyBuf(uint8_t*&curPtr, const void* data, size_t size){
    memcpy(curPtr, data, size);
    curPtr += size;
}

inline void readKeyBuf(uint8_t*&curPtr, void* data, size_t size){
    memcpy(data, curPtr, size);
    curPtr += size;
}

inline void getKeyBuf_cuda(uint8_t*&startPtr, uint8_t *&curPtr, size_t bufSize)
{
    cudaMalloc(&startPtr, bufSize);
    curPtr = startPtr;
}
inline void writeKeyBuf_cuda(uint8_t*&curPtr, const void* data, size_t size){
    cudaMemcpy(curPtr, data, size, cudaMemcpyDeviceToDevice);
    curPtr += size;
}

inline void readKeyBuf_cuda(uint8_t*&curPtr, void* data, size_t size){
    cudaMemcpy(data, curPtr, size, cudaMemcpyDeviceToDevice);
    curPtr += size;
}
// 使用 RowMajor 存储格式
using MatrixRowMajor = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct MM_Keys {
    MatrixRowMajor R_A;
    MatrixRowMajor R_B;
    MatrixRowMajor R_AB;

    MM_Keys(int rowsA, int colsA, int colsB) : R_A(rowsA, colsA), R_B(colsA, colsB), R_AB(rowsA, colsB) {
        R_A.setZero();
        R_B.setZero();
        R_AB.setZero(); 
    }  

    void encode(uint8_t*& buf){
        writeKeyBuf(buf, R_A.data(), R_A.size() * sizeof(int64_t));
        writeKeyBuf(buf, R_B.data(), R_B.size() * sizeof(int64_t));
        writeKeyBuf(buf, R_AB.data(), R_AB.size() * sizeof(int64_t));
    }

    void decode(uint8_t*& buf){
        readKeyBuf(buf, R_A.data(), R_A.size() * sizeof(int64_t));
        readKeyBuf(buf, R_B.data(), R_B.size() * sizeof(int64_t));
        readKeyBuf(buf, R_AB.data(), R_AB.size() * sizeof(int64_t));
    }
};

typedef unsigned char *DCF_Keys;

struct MSB_Keys {
        DCF_Keys k;
        int64_t* random;
        bool* r_msb;
        int N;
        int maxlayer;
        MSB_Keys(int N,int maxlayer=64) : k(nullptr), random(nullptr), r_msb(nullptr), N(N), maxlayer(maxlayer) {
            k = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
            random = (int64_t*)malloc(N * sizeof(int64_t));
            r_msb = (bool*)malloc(N * sizeof(bool));
        }

        void encode(uint8_t*& buf){
            writeKeyBuf(buf, &N, sizeof(int));
            writeKeyBuf(buf, &maxlayer, sizeof(int));
            writeKeyBuf(buf, k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            writeKeyBuf(buf, random, N * sizeof(int64_t));
            writeKeyBuf(buf, r_msb, N * sizeof(bool));
        }

        void decode(uint8_t*& buf){
            readKeyBuf(buf, &N, sizeof(int));
            readKeyBuf(buf, &maxlayer, sizeof(int));
            readKeyBuf(buf, k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            readKeyBuf(buf, random, N * sizeof(int64_t));
            readKeyBuf(buf, r_msb, N * sizeof(bool));
        }
};

//unified MSB keys for GPU and CPU
struct U_MSB_Keys {
        DCF_Keys k;
        int64_t* random;
        bool* r_msb;
        int N;
        int maxlayer;
        U_MSB_Keys(int N,int maxlayer=64) : k(nullptr), random(nullptr), r_msb(nullptr), N(N), maxlayer(maxlayer) {
            cudaMallocManaged(&k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            cudaMallocManaged(&random, N * sizeof(int64_t));
            cudaMallocManaged(&r_msb, N * sizeof(bool));
        }

        void encode(uint8_t*& buf){
            writeKeyBuf(buf, &N, sizeof(int));
            writeKeyBuf(buf, &maxlayer, sizeof(int));
            writeKeyBuf(buf, k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            writeKeyBuf(buf, random, N * sizeof(int64_t));
            writeKeyBuf(buf, r_msb, N * sizeof(bool));
        }

        void decode(uint8_t*& buf){
            readKeyBuf(buf, &N, sizeof(int));
            readKeyBuf(buf, &maxlayer, sizeof(int));
            readKeyBuf(buf, k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            readKeyBuf(buf, random, N * sizeof(int64_t));
            readKeyBuf(buf, r_msb, N * sizeof(bool));
        }
};

struct LUT_Keys {
        DCF_Keys k;
        int64_t* random_in;
        uint64_t* random_out;
        int64_t* random_out_A;
        int N;
        int maxlayer;
        LUT_Keys(int N,int maxlayer=1) : k(nullptr), random_in(nullptr), random_out(nullptr), N(N), maxlayer(maxlayer) {
            k = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
            random_in = (int64_t*)malloc(N * sizeof(int64_t));
            random_out = (uint64_t*)malloc(N * sizeof(uint64_t));
            random_out_A = (int64_t*)malloc(64 * N * sizeof(int64_t));
        }

        void encode(uint8_t*& buf){
            writeKeyBuf(buf, &N, sizeof(int));
            writeKeyBuf(buf, &maxlayer, sizeof(int));
            writeKeyBuf(buf, k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            writeKeyBuf(buf, random_in, N * sizeof(int64_t));
            writeKeyBuf(buf, random_out, N * sizeof(uint64_t));
            writeKeyBuf(buf, random_out_A, 64 * N * sizeof(int64_t));
        }

        void decode(uint8_t*& buf){
            readKeyBuf(buf, &N, sizeof(int));
            readKeyBuf(buf, &maxlayer, sizeof(int));
            readKeyBuf(buf, k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            readKeyBuf(buf, random_in, N * sizeof(int64_t));
            readKeyBuf(buf, random_out, N * sizeof(uint64_t));
            readKeyBuf(buf, random_out_A, 64 * N * sizeof(int64_t));
        }
};

struct LUT_Selection_Keys {
        uint128_t* k;
        int64_t* random_in;
        int64_t* random_out;
        int N;
        int maxlayer;
        LUT_Selection_Keys(int N,int maxlayer=1) : k(nullptr), random_in(nullptr), random_out(nullptr), N(N), maxlayer(maxlayer) {
            k = (uint128_t*)malloc(2 * N * sizeof(uint128_t));
            random_in = (int64_t*)malloc(N * sizeof(int64_t));
            random_out = (int64_t*)malloc(3 * N * sizeof(int64_t));
        }

        void encode(uint8_t*& buf){
            writeKeyBuf(buf, &N, sizeof(int));
            writeKeyBuf(buf, &maxlayer, sizeof(int));
            writeKeyBuf(buf, k, 2 * N * sizeof(uint128_t));
            writeKeyBuf(buf, random_in, N * sizeof(int64_t));
            writeKeyBuf(buf, random_out, 3 * N * sizeof(int64_t));
        }

        void decode(uint8_t*& buf){
            readKeyBuf(buf, &N, sizeof(int));
            readKeyBuf(buf, &maxlayer, sizeof(int));
            readKeyBuf(buf, k, 2 * N * sizeof(uint128_t));
            readKeyBuf(buf, random_in, N * sizeof(int64_t));
            readKeyBuf(buf, random_out, 3 * N * sizeof(int64_t));
        }
};

struct MSB_Keys_cuda {
        DCF_Keys k;
        int64_t* random;
        bool* r_msb;
        int N;
        int maxlayer;
        MSB_Keys_cuda(int N,int maxlayer=64) : k(nullptr), random(nullptr), r_msb(nullptr), N(N), maxlayer(maxlayer) {
            cudaMalloc(&k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            cudaMalloc(&random, N * sizeof(int64_t));
            cudaMalloc(&r_msb, N * sizeof(bool));
        }

        void encode(uint8_t*& buf){
            writeKeyBuf_cuda(buf, &N, sizeof(int));
            writeKeyBuf_cuda(buf, &maxlayer, sizeof(int));
            writeKeyBuf_cuda(buf, k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            writeKeyBuf_cuda(buf, random, N * sizeof(int64_t));
            writeKeyBuf_cuda(buf, r_msb, N * sizeof(bool));
        }

        void decode(uint8_t*& buf){
            readKeyBuf_cuda(buf, &N, sizeof(int));
            readKeyBuf_cuda(buf, &maxlayer, sizeof(int));
            readKeyBuf_cuda(buf, k, N * (1 + 16 + 1 + 18 * maxlayer + 16));
            readKeyBuf_cuda(buf, random, N * sizeof(int64_t));
            readKeyBuf_cuda(buf, r_msb, N * sizeof(bool));
        }
};