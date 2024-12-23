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
using namespace std;
using namespace Eigen;
#define BENCH 10
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

        ~cuda_mpc_core(){
            // delete[] io;
            // delete[] pool;
            delete GMW_A;
            delete GMW_B;
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

        memset(res,0,N*sizeof(bool)); 
        int NN = N;
        
        for(int j = 0; j < BENCH; j++){
            NN = N;
            auto start3 = emp::clock_start();
            for(int i = 0; i < log2(NN); i++){
                NN = NN/2;
                Map<MatrixRowMajor> value_map(value, NN, 1);
                Map<MatrixRowMajor> random_map(keys.random, NN, 1);
                MatrixRowMajor t = value_map - random_map;
                GMW_A->open_vec_2PC(t.data(), t.data(), t.size());
                cudamsbeval_buf(res,k_device,t.data(),r_msb_device, NN, keys.maxlayer, party);
                GMW_B->open_vec(res,res,NN);
            }
            double timeused3 = emp::time_from(start3);
            std::cout << "msb eval Time taken: " << timeused3 / (1000) << " ms" << std::endl;
        }
        

        // auto start = emp::clock_start();
        // for(int i = 0; i < BENCH; i++){
        //     auto start = emp::clock_start();
        //     Map<MatrixRowMajor> value_map(value, N, 1);
        //     Map<MatrixRowMajor> random_map(keys.random, N, 1);
        //     MatrixRowMajor t = value_map - random_map;
        //     GMW_A->open_vec_2PC(t.data(), t.data(), t.size());
        //     cudamsbeval_buf(res,k_device,t.data(),r_msb_device, N, keys.maxlayer, party);
        //     GMW_B->open_vec(res,res,N);
        //     double timeused = emp::time_from(start);
        //     std::cout << "msb eval Time taken: " << timeused / (1000) << " ms" << std::endl;
        // }
        // double timeused = emp::time_from(start);
        // std::cout << "msb eval Time taken: " << timeused / (1000*BENCH) << " ms" << std::endl;

        #ifdef Debug
        auto start2 = emp::clock_start();
        GMW_A->open_vec_2PC(value, value, N);
        double timeused2 = emp::time_from(start2);
        std::cout << "open vec Time taken: " << timeused2 / (1000) << " ms" << std::endl;
        for(int i = 0; i < N; i++){
            if((value[i] < 0) != res[i]){
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


    void cuda_ranksort(int64_t *res, MSB_Keys keys, int64_t *value, int N, int party){

        int maxlayer = keys.maxlayer;

        DCF_Keys k_device;
	    cudaMalloc(&k_device, N * N * (1 + 16 + 1 + 18 * maxlayer + 16));
	    cudaMemcpy(k_device, keys.k, N * N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	    bool* r_msb_device;
	    cudaMalloc(&r_msb_device, N * N * sizeof(bool));
	    cudaMemcpy(r_msb_device, keys.r_msb, N * N * sizeof(bool), cudaMemcpyHostToDevice);

        auto start = emp::clock_start();
        bool* rank_res;
        cudaMallocManaged(&rank_res, N * N * sizeof(bool));

        // map value to matrix
        Map<MatrixRowMajor> value_map(value, 1, N);
        MatrixRowMajor value_mat = value_map.replicate(N, 1);
        MatrixRowMajor value_mat_transpose = value_mat.transpose();
        value_mat = value_mat - value_mat_transpose;
        Map<MatrixRowMajor> random_map(keys.random, N, N);
        MatrixRowMajor t = value_mat - random_map;
        GMW_A->open_vec_2PC(t.data(), t.data(), t.size());
        cudamsbeval_buf(rank_res,k_device,t.data(),r_msb_device, N*N, keys.maxlayer, party);

        int64_t* rank_res_int64 = new int64_t[N*N];
        GMW_B->open_vec_2PC(rank_res_int64, rank_res, N*N);

        Map<MatrixRowMajor> rank_res_map(rank_res_int64, N, N);

        int64_t* rank = new int64_t[N];
        for (int i = 0; i < N; ++i) {
            rank[i] = rank_res_map.row(i).sum();
            res[rank[i]] = value[i];
        }
        double timeused = emp::time_from(start);
        std::cout << "Rank Sort Time taken: " << timeused / (1000) << " ms" << std::endl;

        cudaFree(k_device);
        cudaFree(r_msb_device);
        cudaFree(rank_res);
        delete[] rank_res_int64;
        delete[] rank;
    }

    void cuda_ranksort(int64_t *res, U_MSB_Keys keys, int64_t *value, int N, int party){

        auto start = emp::clock_start();

        bool* rank_res;
        cudaMallocManaged(&rank_res, N * N * sizeof(bool));
        // map value to matrix
        Map<MatrixRowMajor> value_map(value, 1, N);
        MatrixRowMajor value_mat = value_map.replicate(N, 1);
        MatrixRowMajor value_mat_transpose = value_mat.transpose();
        value_mat = value_mat - value_mat_transpose;
        Map<MatrixRowMajor> random_map(keys.random, N, N);
        MatrixRowMajor t = value_mat - random_map;
        GMW_A->open_vec_2PC(t.data(), t.data(), t.size());
        cudamsbeval_buf(rank_res,keys.k,t.data(),keys.r_msb, N*N, keys.maxlayer, party);

        int64_t* rank_res_int64 = new int64_t[N*N];
        GMW_B->open_vec_2PC(rank_res_int64, rank_res, N*N);

        Map<MatrixRowMajor> rank_res_map(rank_res_int64, N, N);

        //Todo: Move to GPU
        int64_t* rank = new int64_t[N];
        for (int i = 0; i < N; ++i) {
            rank[i] = rank_res_map.row(i).sum();
            res[rank[i]] = value[i];
        }
        double timeused = emp::time_from(start);
        std::cout << "Rank Sort Time taken: " << timeused / (1000) << " ms" << std::endl;

        cudaFree(rank_res);
        delete[] rank_res_int64;
        delete[] rank;



    }

    void cuda_ranksort(vector<int64_t>& res, U_MSB_Keys keys, vector<int64_t> value, int party){

        // auto start = emp::clock_start();
        size_t N = value.size();
        bool* rank_res;
        cudaMallocManaged(&rank_res, N * N * sizeof(bool));
        // map value to matrix
        Map<MatrixRowMajor> value_map(value.data(), 1, N);
        MatrixRowMajor value_mat = value_map.replicate(N, 1);
        MatrixRowMajor value_mat_transpose = value_mat.transpose();
        value_mat = value_mat - value_mat_transpose;
        Map<MatrixRowMajor> random_map(keys.random, N, N);
        MatrixRowMajor t = value_mat - random_map;
        GMW_A->open_vec_2PC(t.data(), t.data(), t.size());
        cudamsbeval_buf(rank_res,keys.k,t.data(),keys.r_msb, N*N, keys.maxlayer, party);

        int64_t* rank_res_int64 = new int64_t[N*N];
        GMW_B->open_vec_2PC(rank_res_int64, rank_res, N*N);

        Map<MatrixRowMajor> rank_res_map(rank_res_int64, N, N);

        //Todo: Move to GPU
        int64_t* rank = new int64_t[N];
        for (size_t i = 0; i < N; ++i) {
            rank[i] = rank_res_map.row(i).sum();
            res[rank[i]] = value[i];
        }
        // double timeused = emp::time_from(start);
        // std::cout << "Rank Sort Time taken: " << timeused / (1000) << " ms" << std::endl;

        cudaFree(rank_res);
        delete[] rank_res_int64;
        delete[] rank;



    }

    void cuda_TopK(vector<int64_t>& result, U_MSB_Keys keys, vector<int64_t> input, int K, int maxIterations, int party) {

        vector<int64_t> arr(input);
        int n = arr.size();
        int k = maxIterations;
        
        while (k > 0) {
            // auto start = emp::clock_start();
            // printf("K: %d\n", K);
            // printf("input size: %ld\n", arr.size());

            int p = std::ceil(std::pow(n, 1.0 / k));
            p = std::min(p, n); // 防止 pivots 数量超过数组大小
            // printf("p: %d\n", p);

            // 选择 pivots
            vector<int64_t> pivots;
            vector<int64_t> pivots_sorted(p-1);

            pivots.insert(pivots.end(), arr.end()-(p-1), arr.end());
            cuda_ranksort(pivots_sorted, keys, pivots, party);

            // 根据 pivots 对数组进行分区
            vector<vector<int64_t>> blocks(p);  
            TopK_Partition(blocks, keys, arr, pivots_sorted, party);

            // double timeused = emp::time_from(start);
            // std::cout << "TopK Partition Time taken: " << timeused / (1000) << " ms" << std::endl;
            // std::vector<std::vector<int>> blocks = partition(arr, pivots);
            // printf("blocks: ");
            // for(size_t i = 0; i < blocks.size(); i++){
            //     printf("[");
            //     printf("%ld", blocks[i].size());
            //     printf("] ");
            // }
            // printf("\n");

            // 统计区块大小，找到包含 TopK 的区块
            size_t total = 0;
            arr.clear();
            for (size_t i = 0; i < blocks.size(); i++) {
                if ((total + blocks[i].size()) >= (size_t)K) {
                    arr.insert(arr.end(), blocks[i].begin(), blocks[i].end());
                    break;
                }
            result.insert(result.end(), blocks[i].begin(), blocks[i].end());
            total += blocks[i].size();
            }
        // 更新 K 和 arr
            K -= total;
            n = arr.size();

        // 如果当前区块大小小于等于 K，直接返回
            if (n <= K) {
                vector<int64_t> arr_sorted(arr.size());
                cuda_ranksort(arr_sorted, keys, arr, party);
                arr_sorted.resize(K); // 保留前 K 个元素
                result.insert(result.end(), arr_sorted.begin(), arr_sorted.end());
                return;
            }

        // 迭代次数减少
            k--;
            // double timeused = emp::time_from(start);
            // std::cout << "TopK internal Time taken: " << timeused / (1000) << " ms" << std::endl;
        }
       
        // 最后一步：排序并返回前 K 个元素
        // printf("result size: %ld\n", arr.size());
        vector<int64_t> arr_sorted(arr.size());
        cuda_ranksort(arr_sorted, keys, arr, party);
        result.insert(result.end(), arr_sorted.end()-K, arr_sorted.end());
    }

    void cuda_Max(int64_t &result, U_MSB_Keys keys, vector<int64_t> input, int party){
        vector<int64_t> arr(input);
        int n = arr.size();
        // bool* rank_res;
        // cudaMallocManaged(&rank_res, n * sizeof(bool));

        while(n > 1){
            n = arr.size() / 2;
            
            Map<MatrixRowMajor> arr1_map(arr.data(), 1, n);
            Map<MatrixRowMajor> arr2_map(arr.data() + n, 1, n);
            // cout << "arr1_map: \n" << arr1_map << endl;
            // cout << "arr2_map: \n" << arr2_map << endl;
            MatrixRowMajor tmp = arr1_map - arr2_map;
            // cout << "arr1_map - arr2_map: \n" << tmp << endl;
            Map<MatrixRowMajor> random_map(keys.random, 1, n);
            // cout << "random_map: \n" << random_map << endl;
            MatrixRowMajor t = tmp - random_map;
            GMW_A->open_vec_2PC(t.data(), t.data(), t.size());
            bool* rank_res = new bool[n];
            cudamsbeval_buf(rank_res,keys.k,t.data(),keys.r_msb, n, keys.maxlayer, party);
            int64_t* rank_res_int64 = new int64_t[n];
            memset(rank_res_int64, 0, n * sizeof(int64_t));
            GMW_B->open_vec_2PC(rank_res_int64, rank_res, n);
            Map<MatrixRowMajor> rank_res_map(rank_res_int64, 1, n);
            // cout << "rank_res_map: \n" << rank_res_map << endl;
 
            vector<int64_t> arr_new(n);
            Map<MatrixRowMajor> arr_new_map(arr_new.data(), 1, n);
            arr_new_map = arr_new_map.array() + arr2_map.array() * rank_res_map.array() + arr1_map.array() * (1 - rank_res_map.array());
            // cout<< "arr_new_map: \n" << arr_new_map << endl;
            if(arr.size() % 2 == 1){
                int64_t last_value = arr[arr.size()-1];
                arr_new.push_back(last_value);
            }
            // cout<< "arr_new_map: \n" << arr_new_map << endl;
            // cout << "arr_new: \n";
            // for(int i = 0; i < arr_new.size(); i++){
            //     cout << arr_new[i] << " ";
            // }
            // cout << endl;
            arr = arr_new;
            n = arr.size();
            delete[] rank_res_int64;
            delete[] rank_res;
            
        }
        
        result = arr[0];
        
    }

    void cuda_TopK_CipherGPT(vector<int64_t>& result, U_MSB_Keys keys, vector<int64_t> input, int K, int party){
        vector<int64_t> arr(input);
        int n = arr.size();

        while (K > 0)
        {
            // auto start = emp::clock_start();
            
            vector<int64_t> pivots;
            pivots.insert(pivots.end(), arr.end()-1, arr.end());
            // delete last elements
            arr.pop_back();
            vector<vector<int64_t>> blocks(pivots.size() + 1); 
            TopK_Partition(blocks, keys, arr, pivots, party);

            size_t total = 0;
            arr.clear();
            for (size_t i = blocks.size() - 1; i >= 0; i--) {
                if (total + blocks[i].size() >= K) {
                    arr.insert(arr.end(), blocks[i].begin(), blocks[i].end());
                    break;
                }
                result.insert(result.end(), blocks[i].begin(), blocks[i].end());
                result.insert(result.end(), pivots.begin(), pivots.end());
                total += blocks[i].size() + pivots.size();
            }
            K -= total;
            n = arr.size();

            // 如果当前区块大小小于等于 K，直接返回
            if (n <= K) {
                vector<int64_t> arr_sorted(arr.size());
                cuda_ranksort(arr_sorted, keys, arr, party);
                arr_sorted.resize(K); // 保留前 K 个元素
                result.insert(result.end(), arr_sorted.begin(), arr_sorted.end());
                return;
            }

            // double timeused = emp::time_from(start);
            // std::cout << "TopK internal Time taken: " << timeused / (1000) << " ms" << std::endl;
        }

        // 最后一步：排序并返回前 K 个元素
        // printf("result size: %ld\n", arr.size());
        vector<int64_t> arr_sorted(arr.size());
        cuda_ranksort(arr_sorted, keys, arr, party);
        result.insert(result.end(), arr_sorted.end()-K, arr_sorted.end());

    }

    private:

    void TopK_Partition(vector<vector<int64_t>> &blocks, U_MSB_Keys keys, vector<int64_t> arr, vector<int64_t> pivots, int party) {

        size_t N = arr.size() * pivots.size();
        bool* rank_res;
        cudaMallocManaged(&rank_res, N * sizeof(bool));

        Map<MatrixRowMajor> value_map(arr.data(), 1, arr.size());
        MatrixRowMajor value_mat = value_map.replicate(pivots.size(), 1);
        Map<Vector<int64_t,Dynamic>> pivots_map(pivots.data(), pivots.size());
        value_mat = value_mat.colwise() - pivots_map;
        // std::cout << "pivots_map: \n" << pivots_map << std::endl;
        // std::cout << "value_mat: \n" << value_mat << std::endl;

        Map<MatrixRowMajor> random_map(keys.random, pivots.size(), arr.size());
        MatrixRowMajor t = value_mat - random_map;
        GMW_A->open_vec_2PC(t.data(), t.data(), t.size());


        // auto start2 = emp::clock_start();
        cudamsbeval_buf(rank_res,keys.k,t.data(),keys.r_msb, N, keys.maxlayer, party);
        // double timeused2 = emp::time_from(start2);
        // std::cout << "TopK Partition MSB Time taken: " << timeused2 / (1000) << " ms" << std::endl;


        int64_t* rank_res_int64 = new int64_t[N];
        GMW_B->open_vec_2PC(rank_res_int64, rank_res, N);

        Map<MatrixRowMajor> rank_res_map(rank_res_int64, pivots.size(), arr.size());
        MatrixRowMajor rank_result = rank_res_map.colwise().sum();
        // std::cout << "rank_result: \n" << rank_result << std::endl;

        for(size_t i = 0; i < arr.size(); i++){
            blocks[rank_result.data()[i]].push_back(arr[i]);
        }
        

        cudaFree(rank_res);
        delete[] rank_res_int64;
    }
};


