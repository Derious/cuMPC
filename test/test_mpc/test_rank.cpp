#include <Eigen/Dense>
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "mpc_cuda/mpc_core.h"
#include "../../MPABY_GMW/GMW_protocol.h"
#include <cuda_runtime.h>
using namespace emp;
using namespace Eigen;
using namespace std;
const static int nP = 2;
int party, port;
// #define BENCH 1


int main(int argc, char** argv) {
 
	parse_party_and_port(argv, &party, &port);
    if(party > nP)return 0;
	printf("party:%d	port:%d\n",party,port);
	
    srand(time(NULL));
	NetIOMP<nP> io(party, port);
	ThreadPool pool(4);	
    PRG prg;
    cuda_mpc_core<nP> *cuda_mpc = new cuda_mpc_core<nP>(&io, &pool, party);
    // double total_time = 0;

    
    uint64_t N = atoi(argv[3]);
    uint64_t P = 15;
    uint64_t bufsize = P*N;

    int K = atoi(argv[4]); // 找出前 10 个元素
    int maxIterations = atoi(argv[5]); // 设置最大迭代次数
    std::vector<int64_t> result;
    
    cout << "N: " << N << endl;
    MSB_Keys keys(bufsize);
    
    auto start = std::chrono::high_resolution_clock::now();
    cuda_mpc->cuda_msb_keygen(keys, bufsize, party);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("MSB keygen Time taken: %f milliseconds\n", elapsed.count() * 1000 / 1);

    //Online Phase
    printf("Online Phase\n");
    uint8_t *startPtr, *curPtr;
    getKeyBuf(startPtr, curPtr, (uint64_t)bufsize*MSB_KEY_SIZE);
    keys.encode(curPtr);

    U_MSB_Keys u_keys(bufsize);
    u_keys.decode(startPtr);
    int64_t* value = new int64_t[N];
    prg.random_data(value, N*sizeof(int64_t));
    for(uint64_t i = 0; i < N; i++){
        value[i] = value[i] >> 20;
        // printf("value[%ld]=%lx\n", i, value[i]);
    }

    vector<int64_t> input(value, value + N);
    // vector<int64_t> pivots(value,value + P);

    //  cuda_mpc->cuda_TopK(result, u_keys, input, K, maxIterations, party);
    cuda_mpc->cuda_TopK_CipherGPT(result, u_keys, input, K, party);
    for(int i = 0; i < 1; i++){
        result.clear();
        auto start = emp::clock_start();
        // cuda_mpc->cuda_TopK(result, u_keys, input, K, maxIterations, party);
        cuda_mpc->cuda_TopK_CipherGPT(result, u_keys, input, K, party);
        double timeused = emp::time_from(start);
        std::cout << "TopK_CipherGPT Time taken: " << timeused / (1000) << " ms" << std::endl;
    }

    cuda_mpc->cuda_TopK(result, u_keys, input, K, maxIterations, party);
    // cuda_mpc->cuda_TopK_CipherGPT(result, u_keys, input, K, party);
    for(int i = 0; i < 1; i++){
        result.clear();
        auto start = emp::clock_start();
        cuda_mpc->cuda_TopK(result, u_keys, input, K, maxIterations, party);
        // cuda_mpc->cuda_TopK_CipherGPT(result, u_keys, input, K, party);
        double timeused = emp::time_from(start);
        std::cout << "TopK Time taken: " << timeused / (1000) << " ms" << std::endl;
    }

    // std::vector<int64_t> result2(N);
    int64_t result2;
    cuda_mpc->cuda_Max(result2, u_keys, input, party);
    for(int i = 0; i < 1; i++){
        auto start = emp::clock_start();
        cuda_mpc->cuda_Max(result2, u_keys, input, party);
        // cuda_mpc->cuda_TopK_CipherGPT(result, u_keys, input, K, party);
        double timeused = emp::time_from(start);
        std::cout << "Max Time taken: " << timeused / (1000) << " ms" << std::endl;
    }

    
    cuda_mpc->GMW_A->open_vec_2PC(result.data(), result.data(), result.size());

    printf("result: ");
    std::sort(result.begin(), result.end());

    cuda_mpc->GMW_A->open_vec_2PC(value, value, N);
    cuda_mpc->GMW_A->open_vec_2PC(&result2, &result2, 1);
    std::sort(value, value + N);    
    for(size_t i = N-K; i < N; i++){
        if(value[i] != result[i - N + K]){
            printf("value[%ld]=%lx, result[%ld]=%lx\n", i, value[i], i - N + K, result[i - N + K]);
        }
        printf("value[%ld]=%lx, result2=%lx\n", N-1, value[N-1], result2);

    }
    printf("\n");


    

    delete cuda_mpc;
    
    
}