#include <stdio.h>
#include <string.h>
#include "aes_cuda.h"
#include "../mpc_keys/uint128_type.h"
#include "../mpc_keys/keys_type.h"
#include "aes_prg_device.h"


__device__ void PRG_cuda(uint32_t *key, uint128_t input, uint128_t& output1, uint128_t& output2, int& bit1, int& bit2){
	input = input.set_lsb_zero();

	uint128_t stash[2];
	stash[0] = input;
	stash[1] = input.reverse_lsb();


    AES_encrypt_cu(stash[0].get_bytes(), stash[0].get_bytes(), key);
    AES_encrypt_cu(stash[1].get_bytes(), stash[1].get_bytes(), key);

	stash[0] = stash[0] ^ input;
	stash[1] = stash[1] ^ input;
	stash[1] = stash[1].reverse_lsb();

	bit1 = stash[0].get_lsb();
	bit2 = stash[1].get_lsb();

	output1 = stash[0].set_lsb_zero();
	output2 = stash[1].set_lsb_zero();
}

__device__ void fss_gen_device(AES_Generator_device* prg, uint32_t *key, uint128_t alpha, int n, DCF_Keys k0, DCF_Keys k1){
	// int maxlayer = n - 7;
	int maxlayer = n;
    const int MAX_LAYER = 64;

	uint128_t s[MAX_LAYER + 1][2];
	int t[MAX_LAYER + 1 ][2];
	uint128_t sCW[MAX_LAYER];
	int tCW[MAX_LAYER][2];

	s[0][0] = prg->random(); 
	s[0][1] = prg->random();
	t[0][0] = s[0][0].get_lsb();
	t[0][1] = t[0][0] ^ 1;
	s[0][0] = s[0][0].set_lsb_zero();
	s[0][1] = s[0][1].set_lsb_zero();

	int i;
	uint128_t s0[2], s1[2]; // 0=L,1=R
	#define LEFT 0
	#define RIGHT 1
	int t0[2], t1[2];
	for(i = 1; i<= maxlayer; i++){
		PRG_cuda(key, s[i-1][0], s0[LEFT], s0[RIGHT], t0[LEFT], t0[RIGHT]);
		PRG_cuda(key, s[i-1][1], s1[LEFT], s1[RIGHT], t1[LEFT], t1[RIGHT]);

		int keep, lose;
        // int alphabit = getbit(alpha, n, i);
        int alphabit = alpha.get_bit(n-i);
		if(alphabit == 0){
			keep = LEFT;
			lose = RIGHT;
		}else{
			keep = RIGHT;
			lose = LEFT;
		}

		sCW[i-1] = s0[lose] ^ s1[lose];

		tCW[i-1][LEFT] = t0[LEFT] ^ t1[LEFT] ^ alphabit ^ 1;
		tCW[i-1][RIGHT] = t0[RIGHT] ^ t1[RIGHT] ^ alphabit;

		if(t[i-1][0] == 1){
			s[i][0] = s0[keep] ^ sCW[i-1];
			t[i][0] = t0[keep] ^ tCW[i-1][keep];
		}else{
			s[i][0] = s0[keep];
			t[i][0] = t0[keep];
		}

		if(t[i-1][1] == 1){
			s[i][1] = s1[keep] ^ sCW[i-1];
			t[i][1] = t1[keep] ^ tCW[i-1][keep];
		}else{
			s[i][1] = s1[keep];
			t[i][1] = t1[keep];
		}
	}

    uint128_t finalblock(0,1);
	finalblock = finalblock ^ s[maxlayer][0];
	finalblock = finalblock ^ s[maxlayer][1];
    // finalblock.print_uint128("finalblock = ", finalblock);

	// unsigned char *buff0;
	// unsigned char *buff1;
	// buff0 = (unsigned char*) malloc(1 + 16 + 1 + 18 * maxlayer + 16);
	// buff1 = (unsigned char*) malloc(1 + 16 + 1 + 18 * maxlayer + 16);

	// if(buff0 == NULL || buff1 == NULL){
	// 	printf("Memory allocation failed\n");
	// 	return;
	// }

	k0[0] = n;
	memcpy(&k0[1], &s[0][0], 16);
	k0[17] = t[0][0];
	for(i = 1; i <= maxlayer; i++){
		memcpy(&k0[18 * i], &sCW[i-1], 16);
		k0[18 * i + 16] = tCW[i-1][0];
		k0[18 * i + 17] = tCW[i-1][1]; 
	}
	memcpy(&k0[18 * maxlayer + 18], &finalblock, 16); 

	k1[0] = n;
	memcpy(&k1[18], &k0[18], 18 * (maxlayer));
	memcpy(&k1[1], &s[0][1], 16);
	k1[17] = t[0][1];
	memcpy(&k1[18 * maxlayer + 18], &finalblock, 16);

	// memcpy(k0, buff0, 1 + 16 + 1 + 18 * maxlayer + 16);
	// memcpy(k1, buff1, 1 + 16 + 1 + 18 * maxlayer + 16);
	// free(buff0);
	// free(buff1);
}

__device__ uint128_t dcf_eval_device(uint32_t *key, DCF_Keys k, uint128_t x){
	int n = k[0];
	int maxlayer = n;
    const int MAX_LAYER = 64;

	uint128_t s[MAX_LAYER + 1];
	int t[MAX_LAYER + 1];
	uint128_t sCW[MAX_LAYER];
	int tCW[MAX_LAYER][2];
	uint128_t finalblock;

	memcpy(&s[0], &k[1], 16);
	t[0] = k[17];

	int i;
	for(i = 1; i <= maxlayer; i++){
		memcpy(&sCW[i-1], &k[18 * i], 16);
		tCW[i-1][0] = k[18 * i + 16];
		tCW[i-1][1] = k[18 * i + 17];
	}

	memcpy(&finalblock, &k[18 * (maxlayer + 1)], 16);

	uint128_t sL, sR;
	uint128_t res(0,0);
	int tL, tR;

    // first layer
    PRG_cuda(key, s[0], sL, sR, tL, tR); 

	sL = sL ^ sCW[0].select(t[0]);
	sR = sR ^ sCW[0].select(t[0]);
	tL = tL ^ (tCW[0][0]*t[0]);
	tR = tR ^ (tCW[0][1]*t[0]);	

	int xbit = x.get_bit(n-1);

    s[1] = sR.select(xbit) ^ sL.select((1-xbit));
    t[1] = tR * xbit + tL * (1-xbit);

    res = res ^ uint128_t(0, xbit*t[0]);

	for(i = 2; i <= maxlayer; i++){
        PRG_cuda(key, s[i - 1], sL, sR, tL, tR); 

		sL = sL ^ sCW[i-1].select(t[i-1]);
		sR = sR ^ sCW[i-1].select(t[i-1]);
		tL = tL ^ (tCW[i-1][0]*t[i-1]);
		tR = tR ^ (tCW[i-1][1]*t[i-1]);	

		int xbit = x.get_bit(n-i);
		s[i] = sR.select(xbit) ^ sL.select((1-xbit));
		t[i] = tR * xbit + tL * (1-xbit);

        int xbit_last = x.get_bit(n-i+1);
        int changed = (xbit_last * (1 - xbit)) | ((1 - xbit_last) * xbit);
        res = res ^ uint128_t(0, changed*t[i-1]);
	}
	xbit = 1-x.get_bit(0);
    res = res ^ uint128_t(0, t[maxlayer]*xbit);
	return res;
}

__global__ void fss_genaeskey_kernel(uint32_t key[4 * (14 + 1)]) {
	// 测试密钥 (16字节 = 128位)
    uint64_t userkey1 = 597349; uint64_t userkey2 = 121379; 
	uint128_t userkey(userkey1, userkey2);
    
    // 扩展密钥
    if (AES_set_encrypt_key_cu(userkey.get_bytes(), 128, key) != 0) {
        printf("Key expansion failed!\n");
        return;
    }
}

__global__ void fss_gen_kernel(uint32_t key[4 * (14 + 1)], uint64_t* alpha, int n,DCF_Keys k0, DCF_Keys k1, int N, int maxlayer) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	AES_Generator_device prg;
	unsigned char* k0_local;
	unsigned char* k1_local;
	uint128_t alpha_tid(0, alpha[tid]);
	k0_local = (unsigned char*) (k0 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	k1_local = (unsigned char*) (k1 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	fss_gen_device(&prg, expanded_key, alpha_tid, n, k0_local, k1_local);
}

__global__ void fss_eval_kernel(bool* res, uint32_t key[4 * (14 + 1)], uint64_t* alpha, int n, DCF_Keys k, int N, int maxlayer) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	unsigned char* k_local = (unsigned char*) (k + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t alpha_tid(0, alpha[tid]);
	res[tid] = dcf_eval_device(expanded_key, k_local, alpha_tid).get_lsb();
}	



__global__ void aes_test_kernel(int N, DCF_Keys k0, DCF_Keys k1) {
    // 测试密钥 (16字节 = 128位)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	
    uint64_t userkey1 = 597349; uint64_t userkey2 = 121379; 
	uint128_t userkey(userkey1, userkey2);
    
    // 扩展密钥
    uint32_t expanded_key[4 * (14 + 1)];  // AES-128需要11组轮密钥
    if (AES_set_encrypt_key_cu(userkey.get_bytes(), 128, expanded_key) != 0) {
        printf("Key expansion failed!\n");
        return;
    }
    AES_Generator_device prg;
    uint64_t random = prg.random().get_low();
    uint64_t random2 = prg.random().get_low();
    uint128_t output1, output2;

	int maxlayer = 64;
	unsigned char* k0_local;
	unsigned char* k1_local;

	k0_local = (unsigned char*) (k0 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	k1_local = (unsigned char*) (k1 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
    fss_gen_device(&prg, expanded_key, uint128_t(0, random), 64, k0_local, k1_local);

    output1 = dcf_eval_device(expanded_key, k0_local, uint128_t(0, random2));
    output2 = dcf_eval_device(expanded_key, k1_local, uint128_t(0, random2));
    uint128_t res = output1 ^ output2;
    printf("random < random2 = %s\n", (random < random2) == res.get_lsb()? "success" : "failed");
}

__global__ void fss_msb_keygen_kernel(uint32_t key[4 * (14 + 1)], DCF_Keys k0, DCF_Keys k1, int64_t* random0, int64_t* random1, bool* r_msb0, bool* r_msb1, int N, int maxlayer){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	AES_Generator_device prg;
	unsigned char* k0_local;
	unsigned char* k1_local;
	k0_local = (unsigned char*) (k0 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	k1_local = (unsigned char*) (k1 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint64_t random0_local = prg.random().get_low();
	uint64_t random1_local = prg.random().get_low();
	uint64_t random_local = random0_local + random1_local;
	// printf("random_local = %lx\n", random_local);
	uint64_t r_prime = ((uint64_t)1 << 63);
	// printf("r_prime = %lx\n", r_prime);
	r_prime = r_prime - (random_local << 1 >> 1);
	// printf("r_prime = %lx\n", r_prime);
	uint128_t random_tid(0, r_prime);
	fss_gen_device(&prg, expanded_key, random_tid, 64, k0_local, k1_local);
	r_msb0[tid] = prg.random().get_lsb();
	r_msb1[tid] = (random_local >> 63) != r_msb0[tid];
	random0[tid] = (int64_t)random0_local;
	random1[tid] = (int64_t)random1_local;
}

__global__ void fss_msb_eval_kernel(bool* res, uint32_t key[4 * (14 + 1)], DCF_Keys k, int64_t* value, bool* r_msb, int N, int maxlayer, int select){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	unsigned char* k_local = (unsigned char*) (k + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	// printf("value[tid] = %lx\n", value[tid]);
	uint64_t value_tid = ((uint64_t)value[tid] << 1) >> 1;
	// printf("value_tid = %lx\n", value_tid);
	uint128_t value_tid_128(0, value_tid);
	bool res_local = dcf_eval_device(expanded_key, k_local, value_tid_128).get_lsb();
	// printf("res_local = %d\n", res_local);
	res_local = res_local != r_msb[tid];
	// printf("res_local = %d\n", res_local);
	bool value_msb = ((uint64_t)value[tid]) >> 63;
	// printf("value_msb = %d\n", value_msb);
	res_local = res_local != select*value_msb;
	// printf("res[tid] = %d\n", res_local);
	res[tid] = res_local;
}

extern "C" void cudamsbkeygen(DCF_Keys k0, DCF_Keys k1, int64_t* random0, int64_t* random1, bool* r_msb0, bool* r_msb1, int N, int maxlayer){

	DCF_Keys k0_device;
	cudaMalloc(&k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	DCF_Keys k1_device;
	cudaMalloc(&k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	int64_t* random0_device;
	cudaMalloc(&random0_device, N * sizeof(int64_t));
	cudaMemcpy(random0_device, random0, N * sizeof(int64_t), cudaMemcpyHostToDevice);

	int64_t* random1_device;
	cudaMalloc(&random1_device, N * sizeof(int64_t));
	cudaMemcpy(random1_device, random1, N * sizeof(int64_t), cudaMemcpyHostToDevice);

	bool* r_msb0_device;
	cudaMalloc(&r_msb0_device, N * sizeof(bool));
	cudaMemcpy(r_msb0_device, r_msb0, N * sizeof(bool), cudaMemcpyHostToDevice);

	bool* r_msb1_device;
	cudaMalloc(&r_msb1_device, N * sizeof(bool));
	cudaMemcpy(r_msb1_device, r_msb1, N * sizeof(bool), cudaMemcpyHostToDevice);

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	fss_msb_keygen_kernel<<<blocks, threads>>>(aes_key, k0_device, k1_device, random0_device, random1_device, r_msb0_device, r_msb1_device, N, maxlayer);

	cudaMemcpy(k0, k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(k1, k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(random0, random0_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(random1, random1_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(r_msb0, r_msb0_device, N * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(r_msb1, r_msb1_device, N * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
	cudaFree(k0_device);
	cudaFree(k1_device);
	cudaFree(random0_device);
	cudaFree(random1_device);
	cudaFree(r_msb0_device);
	cudaFree(r_msb1_device);
	cudaFree(aes_key);
}

extern "C" void cudamsbeval(bool* res, DCF_Keys k, int64_t* value, bool* r_msb, int N, int maxlayer, int party){

	//move MSB_keys to device
	DCF_Keys k_device;
	cudaMalloc(&k_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	cudaMemcpy(k_device, k, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	int64_t* value_device;
	cudaMalloc(&value_device, N * sizeof(int64_t));
	cudaMemcpy(value_device, value, N * sizeof(int64_t), cudaMemcpyHostToDevice);

	bool* r_msb_device;
	cudaMalloc(&r_msb_device, N * sizeof(bool));
	cudaMemcpy(r_msb_device, r_msb, N * sizeof(bool), cudaMemcpyHostToDevice);

	bool* res_device;
	cudaMalloc(&res_device, N * sizeof(bool));

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 512 ? 512 : N;
	int blocks = (N + threads - 1) / threads;
	int select = party == 1 ? 1 : 0;

	// cudaEvent_t start1, stop1;
    // cudaEventCreate(&start1);
    // cudaEventCreate(&stop1);

	// cudaEventRecord(start1);
	fss_msb_eval_kernel<<<blocks, threads>>>(res_device, aes_key, k_device, value_device, r_msb_device, N, maxlayer, select);
	// cudaEventRecord(stop1);

	// cudaEventSynchronize(stop1);
	// float time1 = 0;
    // cudaEventElapsedTime(&time1, start1, stop1);
    // printf("msb eval Kernel Time taken: %.3f ms\n", time1);

	cudaMemcpy(res, res_device, N * sizeof(bool), cudaMemcpyDeviceToHost);


	// 等待GPU完成
    cudaDeviceSynchronize();
	cudaFree(k_device);
	cudaFree(value_device);
	cudaFree(r_msb_device);
	cudaFree(res_device);
	cudaFree(aes_key);
}

extern "C" void cudafsskeygen(DCF_Keys k0, DCF_Keys k1, uint64_t* alpha, int N, int n,int maxlayer){

	DCF_Keys k0_device;
	cudaMalloc(&k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	DCF_Keys k1_device;
	cudaMalloc(&k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	uint64_t* alpha_device;
	cudaMalloc(&alpha_device, N * sizeof(uint64_t));
	cudaMemcpy(alpha_device, alpha, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	fss_gen_kernel<<<blocks, threads>>>(aes_key, alpha_device, n, k0_device, k1_device, N, maxlayer);

	cudaMemcpy(k0, k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(k1, k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);	

	cudaFree(k0_device);
	cudaFree(k1_device);
	cudaFree(alpha_device);
	cudaFree(aes_key);
}

extern "C" void cudafsseval(bool *res, DCF_Keys key, uint64_t *value,int N, int maxlayer, int party){

	DCF_Keys key_device;
	cudaMalloc(&key_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	cudaMemcpy(key_device, key, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	uint64_t* value_device;
	cudaMalloc(&value_device, N * sizeof(uint64_t));
	cudaMemcpy(value_device, value, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

	bool* res_device;
	cudaMalloc(&res_device, N * sizeof(bool));

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;	
    fss_eval_kernel<<<blocks, threads>>>(res_device, aes_key, value_device, 64, key_device, N, maxlayer);   

	cudaMemcpy(res, res_device, N * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaFree(key_device);
	cudaFree(value_device);
}

extern "C" int test_dcf() {
    // 启动kernel
	DCF_Keys k0;
	DCF_Keys k1;
	int maxlayer = 64;
	int N = 1000;
	cudaMalloc(&k0, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	cudaMalloc(&k1, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	bool* res1;
	cudaMalloc(&res1, N * sizeof(bool));
	bool* res2;
	cudaMalloc(&res2, N * sizeof(bool));

	bool* res1_host;
	cudaMallocHost(&res1_host, N * sizeof(bool));
	bool* res2_host;
	cudaMallocHost(&res2_host, N * sizeof(bool));

	// uint128_t alpha1 = uint128_t(0, 1);
	// uint128_t alpha2 = uint128_t(0, 2);

	uint64_t* alpha1_host;
	cudaMallocHost(&alpha1_host, N * sizeof(uint64_t));
	uint64_t* alpha2_host;
	cudaMallocHost(&alpha2_host, N * sizeof(uint64_t));
	for(int i = 0; i < N; i++){
		alpha1_host[i] = i+1;
		alpha2_host[i] = i+3;
	}

	uint64_t* value1;
	cudaMalloc(&value1, N * sizeof(uint64_t));
	uint64_t* value2;
	cudaMalloc(&value2, N * sizeof(uint64_t));
	cudaMemcpy(value1, alpha1_host, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(value2, alpha2_host, N * sizeof(uint64_t), cudaMemcpyHostToDevice);


	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

    // 创建CUDA事件
    cudaEvent_t start1, stop1, start2, stop2, start3, stop3;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    
    // 测量第一个kernel: fss_gen_kernel
    cudaEventRecord(start1);
    fss_gen_kernel<<<blocks, threads>>>(aes_key, value1, 64, k0, k1, N, maxlayer);
    cudaEventRecord(stop1);
    
    // 测量第二个kernel: first fss_eval_kernel
    cudaEventRecord(start2);
    fss_eval_kernel<<<blocks, threads>>>(res1, aes_key, value2, 64, k0, N, maxlayer);
	cudaMemcpy(res1_host, res1, N * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop2);
    
    // 测量第三个kernel: second fss_eval_kernel
    cudaEventRecord(start3);
    fss_eval_kernel<<<blocks, threads>>>(res2, aes_key, value2, 64, k1, N, maxlayer);
	cudaMemcpy(res2_host, res2, N * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop3);
    
    // 同步并获取时间
    cudaEventSynchronize(stop1);
    cudaEventSynchronize(stop2);
    cudaEventSynchronize(stop3);
    
    float time1 = 0, time2 = 0, time3 = 0;
    cudaEventElapsedTime(&time1, start1, stop1);
    cudaEventElapsedTime(&time2, start2, stop2);
    cudaEventElapsedTime(&time3, start3, stop3);
    
    // 打印结果
    printf("fss_gen_kernel time: %.3f ms\n", time1);
    printf("First fss_eval_kernel time: %.3f ms\n", time2);
    printf("Second fss_eval_kernel time: %.3f ms\n", time3);
    
    // 销毁事件
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
	
	for(int i = 0; i < N; i++){
		bool res = res1_host[i] ^ res2_host[i];
		printf("res = %d\n", res);
	}
    
    
    // 等待GPU完成
    cudaDeviceSynchronize();

	cudaFree(k0);
	cudaFree(k1);
	cudaFree(res1);
	cudaFree(res2);
		// cudaFree(alpha1_host);
		// cudaFree(alpha2_host);
    
    // 检查错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}