#pragma once
#include <Eigen/Dense>
using namespace Eigen;

#define MSB_KEY_SIZE 1203

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