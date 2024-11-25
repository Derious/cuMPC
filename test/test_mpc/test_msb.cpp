#include <Eigen/Dense>
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "mpc_cuda/mpc_core.h"
#include "../../MPABY_GMW/GMW_protocol.h"
using namespace emp;
using namespace Eigen;
using namespace std;
const static int nP = 2;
int party, port;
#define BENCH 20


int main(int argc, char** argv) {
 
	parse_party_and_port(argv, &party, &port);
    if(party > nP)return 0;
	printf("party:%d	port:%d\n",party,port);
	

	NetIOMP<nP> io(party, port);
	ThreadPool pool(4);	
    PRG prg;
    cuda_mpc_core<nP> *cuda_mpc = new cuda_mpc_core<nP>(&io, &pool, party);
    
    int N = 30520;
    MSB_Keys keys(N);
    cudaWarmup(1024 * 1024, 1);
    auto start = clock_start();
    cuda_mpc->cuda_msb_keygen(keys, N, party);
    double timeused = time_from(start);
    cout << "MSB keygen time used: " << timeused / 1000 << " ms" << endl;

    int64_t* value = new int64_t[N];
    prg.random_data(value, N*sizeof(int64_t));


    bool* res = new bool[N];
    start = clock_start();
    for(int i = 0; i < BENCH; i++){
        cuda_mpc->cuda_msb_eval(res, keys, value, N, party);
    }
    timeused = time_from(start);
    cout << "MSB eval time used: " << timeused / BENCH / 1000 << " ms" << endl;


    cuda_mpc->GMW_B->open_vec(res, res, N);
    cuda_mpc->GMW_A->open_vec(value, value, N);
    for(int i = 0; i < N; i++){
        if((value[i] >= 0) == res[i]){
            printf("Error: value[%d]=%ld, res[%d]=%d\n", i, value[i], i, res[i]);
        }
    }    
    delete[] value;
    delete[] res;
    delete cuda_mpc;
}