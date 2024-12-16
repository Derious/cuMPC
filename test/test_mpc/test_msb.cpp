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
	

	NetIOMP<nP> io(party, port);
	ThreadPool pool(4);	
    PRG prg;
    cuda_mpc_core<nP> *cuda_mpc = new cuda_mpc_core<nP>(&io, &pool, party);
    // double total_time = 0;

    
    uint64_t N = 128*3072;
    uint64_t bufsize = 256*3072;
    cout << "N: " << N << endl;
    MSB_Keys keys(bufsize);
    cudaWarmup(1024 * 1024, party);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++){
        cuda_mpc->cuda_msb_keygen(keys, bufsize, party);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("MSB keygen Time taken: %f milliseconds\n", elapsed.count() * 1000 / 1);

    //Online Phase
    printf("Online Phase\n");
    // for (size_t t = 0; t < log2(N); t++)
    // {
    // N = N/2;
    int64_t* value = new int64_t[N];
    prg.random_data(value, N*sizeof(int64_t));


    bool* res = new bool[N];

    cuda_mpc->cuda_msb_eval_buf(res, keys, value, N, party);

    delete cuda_mpc;
    
    
}