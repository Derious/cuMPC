#pragma once

#include <Eigen/Dense>
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "../MPABY_GMW/GMW_protocol.h"
#include "../mpc_keys/keys_type.h"
#include "../mpc_keys/aes_prg_host.h"
#include "../mpc_keys/fss_keygen.h"
#include <cuda_runtime.h>
// using namespace emp;
using namespace Eigen;
#define BENCH 20
// #define Debug

// CUDA 函数声明
extern "C" void cudaMatrixMultiply(const int64_t* A, const int64_t* B, int64_t* C, int rowsA, int colsA, int colsB);
extern "C" void cudaMatrixLinear(const int64_t a, const int64_t* A, const int64_t b, const int64_t* B, const int64_t c,int64_t* C, int rowsA, int colsA);
extern "C" void cudaMPC_MM(MatrixRowMajor &Public_C, MatrixRowMajor &Public_A, MatrixRowMajor &Public_B, MM_Keys &keys, int party);
extern "C" void cudaWarmup(int size, int party);
extern "C" int test_dcf();
extern "C" void test_circular_shift(int64_t* output_h, int64_t* input_h, int shift);
extern "C" void cudafsseval(bool *res, DCF_Keys key, uint64_t *value, int N, int maxlayer, int party);
extern "C" void cudafsskeygen(DCF_Keys k0, DCF_Keys k1, uint64_t* alpha, int N, int n, int maxlayer);
extern "C" void cudamsbkeygen(DCF_Keys k0, DCF_Keys k1, int64_t* random0, int64_t* random1, bool* r_msb0, bool* r_msb1, int N, int maxlayer);
extern "C" void cudamsbeval(bool* res, DCF_Keys k, int64_t* value, bool* r_msb, int N, int maxlayer, int party);
extern "C" void cudamsbeval_buf(bool* res, DCF_Keys k_device, int64_t* value, bool* r_msb_deivce, int N, int maxlayer, int party);
extern "C" void cudaluteval(int64_t *res, DCF_Keys key, uint64_t *value,int N, int maxlayer, int party);
extern "C" void cudalutkeygen(DCF_Keys k0, DCF_Keys k1, uint64_t* alpha, int N, int n,int maxlayer);
extern "C" void cudalutevalall(int64_t *res, DCF_Keys key,int* shift, int N, int maxlayer, int party);
extern "C" void cudalutPackkeygen(DCF_Keys k0, DCF_Keys k1, uint64_t* alpha, int N);
extern "C" void cudalutPackevalall(int64_t *res, DCF_Keys key,int* shift, int N, int party);    
extern "C" void LUTkeygen(DCF_Keys k0, DCF_Keys k1, int64_t* random_in0, int64_t* random_in1, uint64_t* random_out0, uint64_t* random_out1, int64_t* random_out0_A, int64_t* random_out1_A, int N);
extern "C" void LUTeval(int64_t *res_device, DCF_Keys keys_device, uint64_t *random_out_device, int64_t *value_device, int N, int maxlayer, int party);
extern "C" void LUTlinear(int64_t* res_shared, int64_t* open_res, int64_t* random_out, int N, int party);
extern "C" void LUTSelectionKeygen(uint128_t* k0_device, uint128_t* k1_device, int64_t* random_in0_device, int64_t* random_in1_device, int64_t* random_out0_device, int64_t* random_out1_device, int N);
extern "C" void LUTSelectionEval(int64_t* res_device, uint128_t* k_device, int64_t* random_out, int64_t* value, int N, int party);
extern "C" void LUTSelectionLinear(int64_t* res_shared, int64_t* Public_value, int64_t* random_out, int N, int party);

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
            io->flush(other_party);
        }
        else{
            io->recv_data(other_party, startPtr, (uint64_t)N*MSB_KEY_SIZE);
            keys.decode(curPtr);
            io->flush(other_party);
        }
        free(startPtr);
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

    void cuda_msb_eval_buf(bool *res, MSB_Keys keys, int64_t *value, int N, int party){

        int maxlayer = keys.maxlayer;

        DCF_Keys k_device;
	    cudaMalloc(&k_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	    cudaMemcpy(k_device, keys.k, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	    bool* r_msb_device;
	    cudaMalloc(&r_msb_device, N * sizeof(bool));
	    cudaMemcpy(r_msb_device, keys.r_msb, N * sizeof(bool), cudaMemcpyHostToDevice);

        // memset(res,0,N*sizeof(bool)); 
        // int NN = N/4;
        // auto start3 = emp::clock_start();
        // for(int j = 0; j < BENCH; j++){
        //     NN = N/4;
        //     for(int i = 0; i < log2(NN); i++){
        //         NN = NN/2;
        //         Map<MatrixRowMajor> value_map(value, NN, 1);
        //         Map<MatrixRowMajor> random_map(keys.random, NN, 1);
        //         MatrixRowMajor t = value_map - random_map;
        //         GMW_A->open_vec_2PC(t.data(), t.data(), t.size());
        //         cudamsbeval_buf(res,k_device,t.data(),r_msb_device, NN, keys.maxlayer, party);
        //         GMW_B->open_vec(res,res,NN);
        //     }
        // }
        // double timeused3 = emp::time_from(start3);
        // std::cout << "msb eval Time taken: " << timeused3 / (1000*BENCH) << " ms" << std::endl;

        auto start = emp::clock_start();
        for(int i = 0; i < BENCH; i++){
            Map<MatrixRowMajor> value_map(value, N, 1);
            Map<MatrixRowMajor> random_map(keys.random, N, 1);
            MatrixRowMajor t = value_map - random_map;
            GMW_A->open_vec_2PC(t.data(), t.data(), t.size());
            cudamsbeval_buf(res,k_device,t.data(),r_msb_device, N, keys.maxlayer, party);
            GMW_B->open_vec(res,res,N);
        }
        double timeused = emp::time_from(start);
        std::cout << "msb eval Time taken: " << timeused / (1000*BENCH) << " ms" << std::endl;

        #ifdef Debug
        auto start2 = emp::clock_start();
        GMW_A->open_vec_2PC(value, value, N);
        double timeused2 = emp::time_from(start2);
        std::cout << "open vec Time taken: " << timeused2 / (1000) << " ms" << std::endl;
        for(int i = 0; i < N; i++){
            if((value[i] >= 0) == res[i]){
                printf("Error: value[%d]=%ld, res[%d]=%d\n", i, value[i], i, res[i]);
            }
        } 
        #endif
    }

    void cuda_lut_keygen(LUT_Keys &keys, int N, int party){

        int other_party = party == 1 ? 2 : 1;
        uint8_t *startPtr, *curPtr;
        getKeyBuf(startPtr, curPtr, (uint64_t)N*LUT_KEY_SIZE);

        if(party == 1){
            //start timer
            auto start = emp::clock_start();
            LUT_Keys keys_other(N);
            LUTkeygen(keys.k, keys_other.k, keys.random_in, keys_other.random_in, keys.random_out, keys_other.random_out, keys.random_out_A, keys_other.random_out_A, N);
            double timeused = emp::time_from(start);
            std::cout << "LUT keygen Time taken: " << timeused / (1000) << " ms" << std::endl;

            keys_other.encode(curPtr);
            io->send_data(other_party, startPtr, (uint64_t)N*LUT_KEY_SIZE);
            io->flush(other_party);
        }
        else{
            io->recv_data(other_party, startPtr, (uint64_t)N*LUT_KEY_SIZE);
            keys.decode(curPtr);
            io->flush(other_party);
        }
        free(startPtr);
    }

    void cuda_lut_eval(int64_t *res, LUT_Keys keys, int64_t *value, int N, int party){

        //key transfer
        auto start_transfer = emp::clock_start();

        uint64_t *random_out_device;
        cudaMalloc(&random_out_device, N * sizeof(uint64_t));
        cudaMemcpy(random_out_device, keys.random_out, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

        int64_t *random_out_A_device;
        cudaMalloc(&random_out_A_device, 64 * N * sizeof(int64_t));
        cudaMemcpy(random_out_A_device, keys.random_out_A, 64 * N * sizeof(int64_t), cudaMemcpyHostToDevice);

        DCF_Keys k_device;
	    cudaMalloc(&k_device, N * (1 + 16 + 1 + 18 * keys.maxlayer + 16));
	    cudaMemcpy(k_device, keys.k, N * (1 + 16 + 1 + 18 * keys.maxlayer + 16), cudaMemcpyHostToDevice);

        int64_t *res_device;
        cudaMalloc(&res_device, N * sizeof(int64_t));

        double timeused_transfer = emp::time_from(start_transfer);  
        std::cout << "Key transfer Time taken: " << timeused_transfer / (1000) << " ms" << std::endl;

        //start timer
        auto start = emp::clock_start();

        Map<MatrixRowMajor> value_map(value, N, 1);
        Map<MatrixRowMajor> random_map(keys.random_in, N, 1);
        MatrixRowMajor t = value_map - random_map;
        GMW_A->open_vec_2PC(t.data(), t.data(), t.size());

        int64_t *t_device;
        cudaMalloc(&t_device, N * sizeof(int64_t));
        cudaMemcpy(t_device, t.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);

        LUTeval(res_device, k_device, random_out_device, t_device, N, keys.maxlayer, party);

        cudaMemcpy(res, res_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);

        GMW_B->openbool_vec_2PC(res,res,N);

        int64_t *openres_device;
        cudaMalloc(&openres_device, N * sizeof(int64_t));
        cudaMemcpy(openres_device, res, N * sizeof(int64_t), cudaMemcpyHostToDevice);

        LUTlinear(res_device, openres_device, random_out_A_device, N, party);

        cudaMemcpy(res, res_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);

        double timeused = emp::time_from(start);
        std::cout << "LUT execution Time taken: " << timeused / (1000) << " ms" << std::endl;


        //test correct
        #ifdef Debug
            // int64_t* r_out = new int64_t[N];
            // GMW_B->openbool_vec_2PC(r_out, (int64_t*)keys.random_out, N);

            // if(party == 1){
            //     for(int i = 0; i < N; i++){
            //         printf("r_out: %ld   res: %ld  r_out^res: %ld\n", r_out[i], res[i], r_out[i]^res[i]);
            //     }
            // }
            GMW_A->open_vec_2PC(res, res, N);
            for(int i = 0; i < N; i++){
                printf("res: %ld\n", res[i]);
            }
        #endif
        // // res = res + random_out - 2*random_out*res
        // Map<MatrixRowMajor> random_out_map(keys.random_out, N, 1);
        // Map<MatrixRowMajor> res_map(res, N, 1);
        // MatrixRowMajor t = res_map + random_out_map - 2 * random_out_map * res_map;
        // GMW_B->open_vec_2PC(t.data(), t.data(), t.size());

        cudaFree(res_device);
        cudaFree(k_device);
        cudaFree(random_out_device);
        cudaFree(openres_device);
        cudaFree(t_device);
        cudaFree(random_out_A_device);
    }


    void cuda_lutSelection_keygen(LUT_Selection_Keys &keys, int N, int party){

        int other_party = party == 1 ? 2 : 1;
        uint8_t *startPtr, *curPtr;
        getKeyBuf(startPtr, curPtr, (uint64_t)N*LUT_KEY_SIZE);



        if(party == 1){
            //start timer
            
            uint128_t *k0_device;
            cudaMalloc(&k0_device, 2 * N * sizeof(uint128_t));
            uint128_t *k1_device;
            cudaMalloc(&k1_device, 2 * N * sizeof(uint128_t));

            int64_t *random_in0_device;
            cudaMalloc(&random_in0_device,  N * sizeof(int64_t));
            int64_t *random_in1_device;
            cudaMalloc(&random_in1_device,  N * sizeof(int64_t));

            int64_t *random_out0_device;
            cudaMalloc(&random_out0_device, 3 * N * sizeof(int64_t));
            int64_t *random_out1_device;
            cudaMalloc(&random_out1_device, 3 * N * sizeof(int64_t));

            LUTSelectionKeygen(k0_device, k1_device, random_in0_device, random_in1_device, random_out0_device, random_out1_device, N);

            cudaMemcpy(keys.k, k0_device, 2 * N * sizeof(uint128_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(keys.random_in, random_in0_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(keys.random_out, random_out0_device, 3 * N * sizeof(int64_t), cudaMemcpyDeviceToHost);

            LUT_Selection_Keys keys_other(N);
            cudaMemcpy(keys_other.k, k1_device, 2 * N * sizeof(uint128_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(keys_other.random_in, random_in1_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(keys_other.random_out, random_out1_device, 3 * N * sizeof(int64_t), cudaMemcpyDeviceToHost);

            keys_other.encode(curPtr);
            io->send_data(other_party, startPtr, (uint64_t)N*LUT_KEY_SIZE);
            io->flush(other_party);

            cudaFree(k0_device);
            cudaFree(k1_device);
            cudaFree(random_in0_device);
            cudaFree(random_in1_device);
            cudaFree(random_out0_device);
            cudaFree(random_out1_device);
        }
        else{
            io->recv_data(other_party, startPtr, (uint64_t)N*LUT_KEY_SIZE);
            keys.decode(curPtr);
            io->flush(other_party);
        }
        free(startPtr);
    }

    void cuda_lutSelection_eval(int64_t *res, LUT_Selection_Keys keys, int64_t *value, int N, int party){

        auto start_transfer = emp::clock_start();
        int64_t *res_correct_device;
        cudaMalloc(&res_correct_device, N * sizeof(int64_t));

        int64_t *res_device;
        cudaMalloc(&res_device, 2 * N * sizeof(int64_t));

        uint128_t *k_device;
        cudaMalloc(&k_device, 2 * N * sizeof(uint128_t));
        cudaMemcpy(k_device, keys.k, 2 * N * sizeof(uint128_t), cudaMemcpyHostToDevice);

        int64_t *random_out_device;
        cudaMalloc(&random_out_device, 3 * N * sizeof(int64_t));
        cudaMemcpy(random_out_device, keys.random_out, 3 * N * sizeof(int64_t), cudaMemcpyHostToDevice);

        double timeused_transfer = emp::time_from(start_transfer);  
        std::cout << "Key transfer Time taken: " << timeused_transfer / (1000) << " ms" << std::endl;

        auto start = emp::clock_start();

        Map<MatrixRowMajor> value_map(value, N, 1);
        Map<MatrixRowMajor> random_map(keys.random_in, N, 1);
        MatrixRowMajor t = value_map - random_map;
        GMW_A->open_vec_2PC(t.data(), t.data(), t.size());

        int64_t *t_device;
        cudaMalloc(&t_device, N * sizeof(int64_t));
        cudaMemcpy(t_device, t.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice);

        LUTSelectionEval(res_device, k_device, random_out_device, t_device, N, party);

        cudaMemcpy(res, res_device, 2 * N * sizeof(int64_t), cudaMemcpyDeviceToHost);

        //For LAN Network Condition
        GMW_A->open_vec_2PC(res, res, N);
        GMW_A->open_vec_2PC(res+N, res+N, N);

        // Map<MatrixRowMajor> res_map(res, 2, N);
        // Map<MatrixRowMajor> random_out_map(keys.random_out, 3, N);
        // MatrixRowMajor res_shared = (party - 1) * res_map.row(0).array() * res_map.row(1).array() - res_map.row(0).array() * random_out_map.row(1).array() - res_map.row(1).array() * random_out_map.row(0).array() + random_out_map.row(2).array();
        // GPU Linear

        int64_t *openres_device;
        cudaMalloc(&openres_device, 2 * N * sizeof(int64_t));
        cudaMemcpy(openres_device, res, 2 * N * sizeof(int64_t), cudaMemcpyHostToDevice);

        LUTSelectionLinear(res_device, openres_device, random_out_device, N, party);

        cudaMemcpy(res, res_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);

        double timeused = emp::time_from(start);
        std::cout << "LUT execution Time taken: " << timeused / (1000) << " ms" << std::endl;

        #ifdef Debug
            GMW_A->open_vec_2PC(res, res, N);
            for(int i = 0; i < N; i++){
                if(party == 1){
                    printf("res: %ld\n", res[i]);
                }
            }
        #endif

        cudaFree(res_correct_device);
        cudaFree(res_device);
        cudaFree(k_device);
        cudaFree(random_out_device);
        cudaFree(t_device);
        cudaFree(openres_device);
    }
};


