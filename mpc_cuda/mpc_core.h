#pragma once

#include <Eigen/Dense>
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "../MPABY_GMW/GMW_protocol.h"
#include "../mpc_keys/keys_type.h"
#include "../mpc_keys/aes_prg_host.h"
#include "../mpc_keys/fss_keygen.h"
// using namespace emp;
using namespace Eigen;


// CUDA 函数声明
extern "C" void cudaMatrixMultiply(const int64_t* A, const int64_t* B, int64_t* C, int rowsA, int colsA, int colsB);
extern "C" void cudaMatrixLinear(const int64_t a, const int64_t* A, const int64_t b, const int64_t* B, const int64_t c,int64_t* C, int rowsA, int colsA);
extern "C" void cudaMPC_MM(MatrixRowMajor &Public_C, MatrixRowMajor &Public_A, MatrixRowMajor &Public_B, MM_Keys &keys, int party);
extern "C" void cudaWarmup(int size, int party);
extern "C" int test_dcf();
extern "C" void cudafsseval(bool *res, DCF_Keys key, uint64_t *value, int N, int maxlayer, int party);
extern "C" void cudafsskeygen(DCF_Keys k0, DCF_Keys k1, uint64_t* alpha, int N, int n, int maxlayer);
extern "C" void cudamsbkeygen(DCF_Keys k0, DCF_Keys k1, int64_t* random0, int64_t* random1, bool* r_msb0, bool* r_msb1, int N, int maxlayer);
extern "C" void cudamsbeval(bool* res, DCF_Keys k, int64_t* value, bool* r_msb, int N, int maxlayer, int party);

template<int nP>
class cuda_mpc_core {

    public:
        NetIOMP<nP> *io;
	    ThreadPool * pool;
        int party;
        PRG prg;
        GMWprotocolA<nP> *GMW_A;
        GMWprotocolB<nP> *GMW_B;
        cuda_mpc_core(NetIOMP<nP> *io, ThreadPool * pool, int party) {
            this->io = io;
            this->pool = pool;
            this->party = party;
            GMW_A = new GMWprotocolA<nP>(io, pool, party);
            GMW_B = new GMWprotocolB<nP>(io, pool, party);
        }

    void CUDA_MPC_MM(MatrixRowMajor &Public_C, MatrixRowMajor &Public_A, MatrixRowMajor &Public_B, MM_Keys &keys, int party) {

        cudaMPC_MM(Public_C, Public_A, Public_B, keys, party);
        GMW_A->open_vec_2PC(Public_C.data(), Public_C.data(), Public_C.size());

    }

    void cuda_Linear(MatrixRowMajor &matC, int64_t a, MatrixRowMajor &matA, int64_t b, MatrixRowMajor &matB, int64_t c) {

        int rowsA = matA.rows();
        int colsA = matA.cols();

        auto start = clock_start();
        cudaMatrixLinear(a, matA.data(), b, matB.data(), c, matC.data(), rowsA, colsA); 
        double timeused  = time_from(start);

        auto start2 = emp::clock_start();
        MatrixRowMajor matC_cpu =(a * matA + b * matB).array() + c;
        double timeused2  = emp::time_from(start2);

        std::cout << "CUDA Time taken: " << timeused / (1000) << " ms" << std::endl;
        std::cout << "Eigen Time taken: " << timeused2 / (1000) << " ms" << std::endl;

}

    void cuda_MM(MatrixRowMajor &matC, MatrixRowMajor matA, MatrixRowMajor matB) {

        int rowsA = matA.rows();
        int colsA = matA.cols();
        int colsB = matB.cols();

        // 调用 CUDA 矩阵乘法
        auto start = emp::clock_start();
        cudaMatrixMultiply(matA.data(), matB.data(), matC.data(), rowsA, colsA, colsB);
        double timeused  = emp::time_from(start);
    

    
        auto start2 = emp::clock_start();
        MatrixRowMajor matC_cpu = matA * matB;
        double timeused2  = emp::time_from(start2);
    
        std::cout << "CUDA Time taken: " << timeused / (1000) << " ms" << std::endl;
        std::cout << "Eigen Time taken: " << timeused2 / (1000) << " ms" << std::endl;

    return ;
}

    void cuda_fss_eval(bool *res, DCF_Keys key, uint64_t *value, int N, int maxlayer, int party){
        cudafsseval(res, key, value, N, maxlayer, party);
        GMW_B->open_vec(res,res,N);
    }

    void cuda_msb_keygen(MSB_Keys &keys, int N, int party){

        int other_party = party == 1 ? 2 : 1;
        uint8_t *startPtr, *curPtr;
        getKeyBuf(startPtr, curPtr, (uint64_t)N*MSB_KEY_SIZE);

        if(party == 1){
            MSB_Keys keys_other(N);
            cudamsbkeygen(keys.k, keys_other.k, keys.random, keys_other.random, keys.r_msb, keys_other.r_msb, N, keys.maxlayer);          
            keys_other.encode(curPtr);
            io->send_data(other_party, startPtr, (uint64_t)N*MSB_KEY_SIZE);
        }
        else{
            io->recv_data(other_party, startPtr, (uint64_t)N*MSB_KEY_SIZE);
            keys.decode(startPtr);
        }
    }

    void cuda_msb_eval(bool *res, MSB_Keys keys, int64_t *value, int N, int party){
        // auto start = emp::clock_start();
        Map<MatrixRowMajor> value_map(value, N, 1);
        Map<MatrixRowMajor> random_map(keys.random, N, 1);
        MatrixRowMajor t = value_map - random_map;
        // double timeused = emp::time_from(start);
        // std::cout << "matrix sub Time taken: " << timeused / (1000) << " ms" << std::endl;

        // auto start2 = emp::clock_start();
        GMW_A->open_vec_2PC(t.data(), t.data(), t.size());
        // double timeused2 = emp::time_from(start2);
        // std::cout << "open vec Time taken: " << timeused2 / (1000) << " ms" << std::endl;

        // auto start3 = emp::clock_start();
        cudamsbeval(res, keys.k, t.data(), keys.r_msb, N, keys.maxlayer, party);
        // double timeused3 = emp::time_from(start3);
        // std::cout << "msb eval Time taken: " << timeused3 / (1000) << " ms" << std::endl;
    }
};


