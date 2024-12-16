#include "../../mpc_cuda/mpc_core.h"
using namespace emp;
using namespace Eigen;
using namespace std;
const static int nP = 2;
int party, port;
#define bench 20


int main(int argc, char** argv) {
 
	parse_party_and_port(argv, &party, &port);
    if(party > nP)return 0;
	printf("party:%d	port:%d\n",party,port);
	

	NetIOMP<nP> io(party, port);
	ThreadPool pool(4);	
    PRG prg;
    cuda_mpc_core<nP> *cuda_mpc = new cuda_mpc_core<nP>(&io, &pool, party);

    cudaWarmup(512, party);
    uint64_t N = 1;
    int64_t* alpha = (int64_t*)malloc(N * sizeof(int64_t));
    for(uint64_t i = 0; i < N; i++){
        if(party == 1){
            alpha[i] = 0;
        }
        else{
            alpha[i] = 0;
        }
    }

    LUT_Selection_Keys keys(N);

    auto start = emp::clock_start();
    cuda_mpc->cuda_lutSelection_keygen(keys, N, party);
    double end = emp::time_from(start);
    printf("Keygen CUDA Time taken: %f milliseconds\n", end/1000);

    //print key size
    uint8_t* startptr, *curptr;
    getKeyBuf(startptr, curptr, N * LUT_KEY_SIZE);
    keys.encode(curptr);
    printf("keys size: %ld\n", (size_t)curptr - (size_t)startptr);

    int64_t* res = new int64_t[2 * N];

    start = emp::clock_start();
    for(int i = 0; i < bench; i++){
        // auto start_eval = emp::clock_start();
        cuda_mpc->cuda_lutSelection_eval(res, keys, alpha, N, party);
        // printf("Eval Time taken: %f milliseconds\n", emp::time_from(start_eval)/1000);
    }
    printf("Time taken: %f milliseconds\n", emp::time_from(start)/1000/bench);


    // int maxlayer = 1;
    // DCF_Keys dcf_k0, dcf_k1;
    // dcf_k0 = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
    // dcf_k1 = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
    // // uint64_t* alpha = new uint64_t[N]; 
    // //N is too large, so we use heap
    // uint64_t* alpha = (uint64_t*)malloc(N * sizeof(uint64_t));
    // int* shift = (int*)malloc(N * sizeof(int));
    // for(uint64_t i = 0; i < N; i++){
    //     alpha[i] = 1;
    //     shift[i] = 2;
    // }

    // auto start = clock_start();
    // cudalutkeygen(dcf_k0, dcf_k1, alpha, N, n, maxlayer);
    // double end = time_from(start);
    // printf("Keygen CUDA Time taken: %f milliseconds\n", end/1000);

    // int64_t* res1 = (int64_t*)malloc(N * 256 * sizeof(int64_t));
    // int64_t* res2 = (int64_t*)malloc(N * 256 * sizeof(int64_t));

    // start = clock_start();
    // for(int i = 0; i < bench; i++){
    //     cudalutevalall(res1, dcf_k0, shift, N, maxlayer, 0);
    //     cudalutevalall(res2, dcf_k1, shift, N, maxlayer, 1);
    // }
    // end = time_from(start);
    // printf("Eval ALL Time taken: %f milliseconds\n", end/1000/bench);

    // auto start = clock_start();
    // cudalutPackkeygen(dcf_k0, dcf_k1, alpha, N);
    // double end = time_from(start);
    // printf("Keygen CUDA Time taken: %f milliseconds\n", end/1000);

    // int64_t* res1 = (int64_t*)malloc(N * sizeof(int64_t));
    // int64_t* res2 = (int64_t*)malloc(N * sizeof(int64_t));

    // auto start_all = clock_start();
    // cudalutPackevalall(res1, dcf_k0, shift, N, 0);
    // for(int i = 0; i < bench; i++){
    //     start = clock_start();
    //     cudalutPackevalall(res2, dcf_k1, shift, N, 1);
    //     end = time_from(start);
    //     printf("Eval ALL Time taken: %f milliseconds\n", end/1000);
    // }
    // end = time_from(start_all);
    // printf("Eval ALL lutpack Time taken: %f milliseconds\n", end/1000);
    
    // // for (size_t i = 0; i < N; i++)
    // // {
    // //     int final_res = res1[i] ^ res2[i];
    // //     // printf("i = %d\n", i);
    // //     printf("final_res = %d\n", final_res);
    // //     printf("\n");
    // // }
    
   


    // free(dcf_k0);
    // free(dcf_k1);
    // delete[] alpha;
    // delete[] res1;
    // delete[] res2;
}

