#include <stdio.h>
#include <string.h>
#include "aes_cuda.h"
#include "../mpc_keys/uint128_type.h"
#include "../mpc_keys/keys_type.h"
#include "aes_prg_device.h"


__device__ const int64_t FIXED_BITS = 20;
__device__ const int64_t FIXED_POINT_SCALE = (1<<FIXED_BITS);

__device__ const float gelu_lut[256] = {
    -0.000009f, -0.000010f, -0.000011f, -0.000013f, -0.000014f, -0.000016f, -0.000017f, -0.000019f,
    -0.000022f, -0.000024f, -0.000027f, -0.000030f, -0.000033f, -0.000037f, -0.000041f, -0.000045f,
    -0.000050f, -0.000055f, -0.000061f, -0.000067f, -0.000074f, -0.000081f, -0.000090f, -0.000099f,
    -0.000109f, -0.000119f, -0.000131f, -0.000144f, -0.000158f, -0.000173f, -0.000189f, -0.000207f,
    -0.000227f, -0.000248f, -0.000270f, -0.000295f, -0.000322f, -0.000350f, -0.000382f, -0.000415f,
    -0.000452f, -0.000491f, -0.000533f, -0.000578f, -0.000627f, -0.000679f, -0.000736f, -0.000796f,
    -0.000861f, -0.000931f, -0.001006f, -0.001086f, -0.001171f, -0.001263f, -0.001360f, -0.001465f,
    -0.001576f, -0.001695f, -0.001821f, -0.001956f, -0.002099f, -0.002252f, -0.002414f, -0.002586f,
    -0.002769f, -0.002962f, -0.003168f, -0.003386f, -0.003616f, -0.003860f, -0.004118f, -0.004391f,
    -0.004678f, -0.004982f, -0.005303f, -0.005641f, -0.005996f, -0.006371f, -0.006765f, -0.007179f,
    -0.007615f, -0.008072f, -0.008552f, -0.009056f, -0.009583f, -0.010136f, -0.010715f, -0.011320f,
    -0.011954f, -0.012615f, -0.013306f, -0.014028f, -0.014780f, -0.015564f, -0.016381f, -0.017232f,
    -0.018117f, -0.019038f, -0.019994f, -0.020988f, -0.022019f, -0.023089f, -0.024198f, -0.025346f,
    -0.026536f, -0.027767f, -0.029039f, -0.030354f, -0.031713f, -0.033115f, -0.034560f, -0.036051f,
    -0.037586f, -0.039166f, -0.040791f, -0.042462f, -0.044179f, -0.045941f, -0.047749f, -0.049603f,
    -0.051502f, -0.053445f, -0.055434f, -0.057467f, -0.059543f, -0.061662f, -0.063824f, -0.066027f,
    -0.068270f, -0.070552f, -0.072873f, -0.075230f, -0.077622f, -0.080048f, -0.082505f, -0.084993f,
    -0.087509f, -0.090050f, -0.092615f, -0.095201f, -0.097806f, -0.100426f, -0.103060f, -0.105703f,
    -0.108353f, -0.111006f, -0.113659f, -0.116308f, -0.118950f, -0.121580f, -0.124195f, -0.126789f,
    -0.129359f, -0.131901f, -0.134408f, -0.136877f, -0.139303f, -0.141680f, -0.144003f, -0.146266f,
    -0.148465f, -0.150593f, -0.152645f, -0.154615f, -0.156496f, -0.158283f, -0.159970f, -0.161549f,
    -0.163015f, -0.164362f, -0.165582f, -0.166670f, -0.167618f, -0.168419f, -0.169068f, -0.169558f,
    -0.169881f, -0.170030f, -0.170000f, -0.169784f, -0.169374f, -0.168764f, -0.167947f, -0.166917f,
    -0.165667f, -0.164191f, -0.162482f, -0.160535f, -0.158342f, -0.155899f, -0.153198f, -0.150235f,
    -0.147003f, -0.143498f, -0.139713f, -0.135644f, -0.131286f, -0.126634f, -0.121684f, -0.116431f,
    -0.110871f, -0.105000f, -0.098815f, -0.092312f, -0.085488f, -0.078341f, -0.070867f, -0.063064f,
    -0.054930f, -0.046464f, -0.037664f, -0.028528f, -0.019055f, -0.009246f, 0.000901f, 0.011386f,
    0.022209f, 0.033368f, 0.044864f, 0.056694f, 0.068859f, 0.081357f, 0.094184f, 0.107341f,
    0.120823f, 0.134628f, 0.148754f, 0.163197f, 0.177954f, 0.193021f, 0.208394f, 0.224069f,
    0.240041f, 0.256307f, 0.272861f, 0.289698f, 0.306814f, 0.324203f, 0.341859f, 0.359776f,
    0.377950f, 0.396373f, 0.415041f, 0.433946f, 0.453083f, 0.472445f, 0.492026f, 0.511818f,
    0.531817f, 0.552014f, 0.572403f, 0.592978f, 0.613731f, 0.634656f, 0.655747f, 0.676996f,
};

// 简单的CUDA核函数
__global__ void warmupKernel(int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int a = 0;
    if (idx < size) {
         a += 1;  // 简单的操作，增加每个元素的值
    }
}

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
// LUT key generation //lut for 8-bit
__device__ void lut_gen_device(AES_Generator_device* prg, uint32_t *key, uint128_t alpha, int n, DCF_Keys k0, DCF_Keys k1){
	// int maxlayer = n - 7;
	int maxlayer = n;
    const int MAX_LAYER = 8;

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

    
	// finalblock = finalblock ^ s[maxlayer][0];
	// finalblock = finalblock ^ s[maxlayer][1];
	int64_t final_value = 1;
	final_value = final_value - s[maxlayer][0].get_low() + s[maxlayer][1].get_low();
	final_value = (1-2*t[maxlayer][1]) * final_value;

	uint128_t finalblock(0,final_value);

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
}

__device__ uint128_t lut_eval_device(uint32_t *key, DCF_Keys k, uint128_t x, int party){
	int n = k[0];
	int maxlayer = n;
    const int MAX_LAYER = 8;

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

	for(i = 1; i <= maxlayer; i++){
        PRG_cuda(key, s[i - 1], sL, sR, tL, tR); 

		sL = sL ^ sCW[i-1].select(t[i-1]);
		sR = sR ^ sCW[i-1].select(t[i-1]);
		tL = tL ^ (tCW[i-1][0]*t[i-1]);
		tR = tR ^ (tCW[i-1][1]*t[i-1]);	

		int xbit = x.get_bit(n-i);
		s[i] = sR.select(xbit) ^ sL.select((1-xbit));
		t[i] = tR * xbit + tL * (1-xbit);
	}
	int64_t res_value = 0;
	res_value = s[maxlayer].get_low() + t[maxlayer]*finalblock.get_low();
	res_value = (1-2*party) * res_value;

	return uint128_t(0, res_value);
}

__device__ uint128_t lut_evalall_device(uint32_t *key, DCF_Keys k, int party){

	int tid = 2*threadIdx.x;
	uint128_t tid_uint128(0, tid);
	int n = k[0];
	int maxlayer = n;
    const int MAX_LAYER = 8;

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
	int tL, tR;

	for(i = 1; i <= maxlayer-1; i++){
        PRG_cuda(key, s[i - 1], sL, sR, tL, tR); 

		sL = sL ^ sCW[i-1].select(t[i-1]);
		sR = sR ^ sCW[i-1].select(t[i-1]);
		tL = tL ^ (tCW[i-1][0]*t[i-1]);
		tR = tR ^ (tCW[i-1][1]*t[i-1]);	

		int xbit = tid_uint128.get_bit(n-i);
		s[i] = sR.select(xbit) ^ sL.select((1-xbit));
		t[i] = tR * xbit + tL * (1-xbit);
	}

	// last layer
	PRG_cuda(key, s[maxlayer-1], sL, sR, tL, tR);
	sL = sL ^ sCW[maxlayer-1].select(t[maxlayer-1]);
	sR = sR ^ sCW[maxlayer-1].select(t[maxlayer-1]);
	tL = tL ^ (tCW[maxlayer-1][0]*t[maxlayer-1]);
	tR = tR ^ (tCW[maxlayer-1][1]*t[maxlayer-1]);	

	int64_t resL = 0;
	resL = sL.get_low() + tL*finalblock.get_low();
	resL = (1-2*party) * resL;

	int64_t resR = 0;
	resR = sR.get_low() + tR*finalblock.get_low();
	resR = (1-2*party) * resR;

	return uint128_t(resL, resR);
}

__device__ void lut_pack_gen_device(AES_Generator_device* prg, uint32_t *key, uint128_t alpha, int n, int maxlayer, DCF_Keys k0, DCF_Keys k1, uint128_t* g_selection_table){

	// int maxlayer = n;
    const int MAX_LAYER = 1;

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

    uint128_t finalblock;
	uint32_t alpha_m = alpha.get_low() % ((uint64_t)1 << (n-maxlayer));
    finalblock = g_selection_table[alpha_m];
	// printf("alpha = %lx\n", alpha.get_low());
	// printf("alpha_m = %x\n", alpha_m);
	// finalblock.print_uint128("finalblock = ", finalblock);
	finalblock = finalblock ^ s[maxlayer][0];
	finalblock = finalblock ^ s[maxlayer][1];
    

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

__device__ void lut_pack_evalall_device(uint128_t &res_L, uint128_t &res_R, uint32_t *key, DCF_Keys k,int maxlayer){

	// int tid = 2*threadIdx.x;
	// uint128_t tid_uint128(0, tid);
	// int n = k[0];
    const int MAX_LAYER = 8;

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
	int tL, tR;

	// for(i = 1; i <= maxlayer-1; i++){
    // PRG_cuda(key, s[0], sL, sR, tL, tR); 

	// sL = sL ^ sCW[0].select(t[0]);
	// sR = sR ^ sCW[0].select(t[0]);
	// tL = tL ^ (tCW[0][0]*t[0]);
	// tR = tR ^ (tCW[0][1]*t[0]);	

		// int xbit = tid_uint128.get_bit(n-i);
		// s[i] = sR.select(xbit) ^ sL.select((1-xbit));
		// t[i] = tR * xbit + tL * (1-xbit);
	// }

	// last layer
	PRG_cuda(key, s[maxlayer-1], sL, sR, tL, tR);
	sL = sL ^ sCW[maxlayer-1].select(t[maxlayer-1]);
	sR = sR ^ sCW[maxlayer-1].select(t[maxlayer-1]);
	tL = tL ^ (tCW[maxlayer-1][0]*t[maxlayer-1]);
	tR = tR ^ (tCW[maxlayer-1][1]*t[maxlayer-1]);	


	res_L = sL^finalblock.select(tL);
	res_R = sR^finalblock.select(tR);


	// return res_L;
}


// FSS generation code
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


__device__ void dcf_pack_gen_device(AES_Generator_device* prg, uint32_t *key, uint128_t alpha, int n, int maxlayer, DCF_Keys k0, DCF_Keys k1, uint128_t* g_prefix_ones_table){

	// int maxlayer = n;
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

    uint128_t finalblock;
	uint32_t alpha_m = alpha.get_low() % ((uint64_t)1 << (n-maxlayer));
    finalblock = g_prefix_ones_table[alpha_m];
	// printf("alpha = %lx\n", alpha.get_low());
	// printf("alpha_m = %x\n", alpha_m);
	// finalblock.print_uint128("finalblock = ", finalblock);
	finalblock = finalblock ^ s[maxlayer][0];
	finalblock = finalblock ^ s[maxlayer][1];
    

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

__device__ uint128_t dcf_pack_eval_device(uint32_t *key, DCF_Keys k, uint128_t x, int maxlayer){

	int n = k[0];
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
	int tL, tR;
	uint64_t res = 0;
	for(i = 1; i <= maxlayer; i++){
		PRG_cuda(key, s[i - 1], sL, sR, tL, tR); 

		sL = sL ^ sCW[i-1].select(t[i-1]);
		sR = sR ^ sCW[i-1].select(t[i-1]);
		tL = tL ^ (tCW[i-1][0]*t[i-1]);
		tR = tR ^ (tCW[i-1][1]*t[i-1]);	

		int xbit = x.get_bit(n-i);
		s[i] = sR.select(xbit) ^ sL.select((1-xbit));
		t[i] = tR * xbit + tL * (1-xbit);
		res = res ^ (tR * (1-xbit));
	}
	// printf("res = %lx\n", res);
	uint128_t eval_res;
	// printf("x = %lx\n", x.get_low());
	// uint32_t x_m = ((uint64_t)x.get_low() << (maxlayer)) >> (maxlayer);
	uint32_t x_m = x.get_low() % ((uint64_t)1 << (n-maxlayer));
	// printf("x_m = %lx\n", x_m);
	// printf("t[maxlayer] = %d\n", t[maxlayer]);
	eval_res = s[maxlayer];
    eval_res = eval_res ^ finalblock.select(t[maxlayer]);
	// eval_res.print_uint128("eval_res = ", eval_res);

	uint32_t tmp = 127 - x_m;
	// printf("tmp = %d\n", tmp);
	eval_res >>= tmp;
	// eval_res.print_uint128("eval_res = ", eval_res);
	tmp = eval_res.get_lsb();
	// printf("tmp = %d\n", tmp);
	res = res ^ tmp;

	return uint128_t(0, res);

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

__global__ void lut_pack_gen_kernel(uint32_t key[4 * (14 + 1)], uint64_t* alpha, int n, DCF_Keys k0, DCF_Keys k1, int N, int maxlayer, uint128_t* g_selection_table) {

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
	lut_pack_gen_device(&prg, expanded_key, alpha_tid, n, maxlayer, k0_local, k1_local, g_selection_table);

}

__global__ void lut_pack_eval_kernel(int64_t* res, uint32_t key[4 * (14 + 1)], DCF_Keys k, int* shift, int N,  int maxlayer) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	unsigned char* k_local = (unsigned char*) (k + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t res_L(0, 0);
	uint128_t res_R(0, 0);
	lut_pack_evalall_device(res_L, res_R, expanded_key, k_local, maxlayer);
	// res[2*tid] = res_L;
	// res[2*tid + 1] = res_R;
	// res_L.print_uint128("res_L = ", res_L);
	// res_R.print_uint128("res_R = ", res_R);
	int pos = shift[tid];
	int target_pos = 0;
	
	int64_t lut_res = 0;
	for(int i = 0; i < 128; i++){
		target_pos = (i + pos) >= 0 ? (i + pos) % 256 : 256 + (i + pos);
		int64_t fixed_val = gelu_lut[target_pos] * FIXED_POINT_SCALE;
		lut_res = lut_res ^ (res_L.get_bit(127-i) * fixed_val);
	}
	for(int i = 0; i < 128; i++){
		target_pos = (i + pos + 128) >= 0 ? (i + pos + 128) % 256 : 256 + (i + pos + 128);
		int64_t fixed_val = gelu_lut[target_pos] * FIXED_POINT_SCALE;
		lut_res = lut_res ^ (res_R.get_bit(127-i) * fixed_val);
	}
	res[tid] = lut_res;
}

__global__ void lut_gen_kernel(uint32_t key[4 * (14 + 1)], uint64_t* alpha, int n,DCF_Keys k0, DCF_Keys k1, int N, int maxlayer) {
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
	lut_gen_device(&prg, expanded_key, alpha_tid, n, k0_local, k1_local);
}

__global__ void lut_eval_kernel(int64_t* res, uint32_t key[4 * (14 + 1)], uint64_t* alpha, DCF_Keys k, int N, int maxlayer, int party) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	unsigned char* k_local = (unsigned char*) (k + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t alpha_tid(0, alpha[tid]);
	res[tid] = lut_eval_device(expanded_key, k_local, alpha_tid, party).get_low();
}	

__global__ void lut_eval_all_kernel(int64_t* res, uint32_t key[4 * (14 + 1)], DCF_Keys k, int* shift, int N, int maxlayer, int party) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	// printf("block_id = %d, thread_id = %d\n", block_id, thread_id);
	if(bid >= N) return;

	// __shared__ uint64_t res_shared[256];
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	unsigned char* k_local = (unsigned char*) (k + bid * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t res_local(0,0);
	res_local = lut_evalall_device(expanded_key, k_local, party);
	// res_shared[2*tid] = res_local.get_high();
	// res_shared[2*tid + 1] = res_local.get_low();
	res[bid * 256 + 2*tid] = res_local.get_high();
	res[bid * 256 + 2*tid + 1] = res_local.get_low();
	// __syncthreads();

	// //shuffle
	// int pos = shift[bid];
	// int target_pos = (2*tid + pos) >= 0 ? (2*tid + pos) % 256 : 256 + (2*tid + pos);
	// int target_pos2 = (2*tid + pos + 1) >= 0 ? (2*tid + pos + 1) % 256 : 256 + (2*tid + pos + 1);
	// res[bid * 256 + 2*tid] = res_shared[target_pos];
	// res[bid * 256 + 2*tid + 1] = res_shared[target_pos2];
}	

__global__ void fss_lut_keygen_kernel(uint32_t key[4 * (14 + 1)], DCF_Keys k0, DCF_Keys k1,
								 int64_t* random_in0, int64_t* random_in1, uint64_t* random_out0, uint64_t* random_out1, 
								 int64_t* random_out0_A, int64_t* random_out1_A, int N, int n, int maxlayer, uint128_t* g_selection_table) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	AES_Generator_device prg;
	unsigned char* k0_local;
	unsigned char* k1_local;
	k0_local = (unsigned char*) (k0 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	k1_local = (unsigned char*) (k1 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));

	int64_t random_in0_local = (uint64_t)prg.random().get_low();
	uint8_t random_in_local = (uint8_t)prg.random().get_low();
	uint64_t random_out0_local = (uint64_t)prg.random().get_low();
	uint64_t random_out1_local = (uint64_t)prg.random().get_low();
	

	int64_t random_in1_local = (int64_t)(random_in_local) - random_in0_local;
	uint128_t random_tid(0, (uint64_t)random_in_local);;

	lut_pack_gen_device(&prg, expanded_key, random_tid, n, maxlayer, k0_local, k1_local, g_selection_table);

	random_in0[tid] = random_in0_local;
	random_in1[tid] = random_in1_local;
	random_out0[tid] = random_out0_local;
	random_out1[tid] = random_out1_local;

	uint64_t random_out = random_out0_local ^ random_out1_local;
	for(int i = 0; i < 64; i++){
		random_out0_A[tid * 64 + i] = prg.random().get_low();
		random_out1_A[tid * 64 + i] = ((random_out >> i) & (uint64_t)1) - random_out0_A[tid * 64 + i];
	}

}

__global__ void fss_lut_eval_kernel(int64_t* res, uint32_t key[4 * (14 + 1)], DCF_Keys k, uint64_t* random_out, int64_t* shift,  int N,  int maxlayer) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	unsigned char* k_local = (unsigned char*) (k + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t res_L(0, 0);
	uint128_t res_R(0, 0);
	lut_pack_evalall_device(res_L, res_R, expanded_key, k_local, maxlayer);
	int64_t pos = shift[tid];
	int64_t target_pos = 0;
	
	int64_t lut_res = 0;
	for(int i = 0; i < 128; i++){
		target_pos = (i + pos) >= 0 ? (i + pos) % 256 : 256 + (i + pos);
		int64_t fixed_val = gelu_lut[target_pos] * FIXED_POINT_SCALE;
		lut_res = lut_res ^ (res_L.get_bit(127-i) * fixed_val);
	}
	for(int i = 0; i < 128; i++){
		target_pos = (i + pos + 128) >= 0 ? (i + pos + 128) % 256 : 256 + (i + pos + 128);
		int64_t fixed_val = gelu_lut[target_pos] * FIXED_POINT_SCALE;
		lut_res = lut_res ^ (res_R.get_bit(127-i) * fixed_val);
	}
	res[tid] = lut_res^random_out[tid];
}

__global__ void fss_lut_linear_kernel(int64_t* res, int64_t* open_res, int64_t* random_out, int N, int party){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	int64_t open_res_local = open_res[tid];
	int64_t random_out_local[64];
	memcpy(random_out_local, random_out + tid * 64, 64 * sizeof(int64_t));
	// res_bit[i] = open_res[i] + random_out[i] - 2*open_res[i]*random_out[i]
	int64_t res_local = 0;
	for(int i = 0; i < 64; i++){
		int64_t res_bit = (open_res_local >> i) & 1;
		res_bit = (party - 1)*res_bit + random_out_local[i] - 2*res_bit*random_out_local[i];
		res_local += res_bit << i;
	}
	res[tid] = res_local;
}

__global__ void fss_lutSelection_keygen_kernel(uint32_t key[4 * (14 + 1)], uint128_t* k0, uint128_t* k1,
								 int64_t* random_in0, int64_t* random_in1, int64_t* random_out0, int64_t* random_out1, 
								 int N, uint128_t* g_selection_table) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	AES_Generator_device prg;

	int64_t random_in0_local = (int64_t)prg.random().get_low();
	uint8_t random_in_local = (uint8_t)prg.random().get_low();
	uint64_t random_out0_local = (uint64_t)prg.random().get_low();
	uint64_t random_out1_local = (uint64_t)prg.random().get_low();

	uint128_t random_a = prg.random();
	uint128_t random_b = prg.random();
	uint128_t random_c = prg.random();

	// c0 = (a0 + a1) * (b0 + b1) - c1
	// c1 = c1
	int64_t C0 = (random_a.get_low() + random_a.get_high()) * (random_b.get_low() + random_b.get_high()) - random_c.get_high();
	int64_t C1 = random_c.get_high();

	uint128_t k0_local_0 = prg.random();
	uint128_t k0_local_1 = prg.random();

	uint128_t T_base = g_selection_table[random_in_local%128];
	int select = (random_in_local >> 7) & (uint8_t)0x1;
	uint128_t T_base_0 = T_base.select(1-select);
	uint128_t T_base_1 = T_base.select(select);
	// printf("random_in_local = %d\n", random_in_local);
	// T_base_0.print_uint128("T_base_0:",T_base_0);
	// T_base_1.print_uint128("T_base_1:",T_base_1);

	uint128_t k1_local_0 = k0_local_0 ^ T_base_0;
	uint128_t k1_local_1 = k0_local_1 ^ T_base_1;

	int64_t random_in1_local = (int64_t)(random_in_local) - random_in0_local;
	uint128_t random_tid(0, (uint64_t)random_in_local);

	k0[2*tid] = k0_local_0;
	k0[2*tid + 1] = k0_local_1;
	k1[2*tid] = k1_local_0;
	k1[2*tid + 1] = k1_local_1;
	random_in0[tid] = random_in0_local;
	random_in1[tid] = random_in1_local;
	random_out0[3*tid] = random_a.get_low();
	random_out0[3*tid + 1] = random_b.get_low();
	random_out0[3*tid + 2] = C0;
	random_out1[3*tid] = random_a.get_high();
	random_out1[3*tid + 1] = random_b.get_high();
	random_out1[3*tid + 2] = C1;

}

__global__ void fss_lutSelection_eval_kernel(int64_t* res, uint128_t* k, int64_t* random_out, int64_t* shift,  int N, int party) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;

	uint128_t res_L = k[2*tid];
	uint128_t res_R = k[2*tid + 1];

	int64_t pos = shift[tid];
	int64_t target_pos = 0;
	// sign = 1 if party = 1, sign = -1 if party = 2
	int sign = 3 - 2*party;
	int64_t res_correct_local = 0;
	int64_t lut_res = 0;
	for(int i = 0; i < 128; i++){
		target_pos = (i + pos) >= 0 ? (i + pos) % 256 : 256 + (i + pos);
		int64_t fixed_val = gelu_lut[target_pos] * FIXED_POINT_SCALE;
		int64_t res_bit =  sign * res_L.get_bit(127-i);
		lut_res = lut_res + (res_bit * fixed_val);
		res_correct_local = res_correct_local + res_bit;
	}
	for(int i = 0; i < 128; i++){
		target_pos = (i + pos + 128) >= 0 ? (i + pos + 128) % 256 : 256 + (i + pos + 128);
		int64_t fixed_val = gelu_lut[target_pos] * FIXED_POINT_SCALE;
		int64_t res_bit = sign * res_R.get_bit(127-i);
		lut_res = lut_res + (res_bit * fixed_val);
		res_correct_local = res_correct_local + res_bit;
	}
	res[2*tid] = lut_res + random_out[3*tid];
	res[2*tid+1] = res_correct_local + random_out[3*tid+1];
}

__global__ void fss_lutSelection_linear_kernel(int64_t* res, int64_t* open_res, int64_t* random_out, int N, int party){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	int64_t Public_A = open_res[2*tid];
	int64_t Public_B = open_res[2*tid+1];
	int64_t Delta_A = random_out[3*tid];
	int64_t Delta_B = random_out[3*tid+1];
	int64_t Delta_C = random_out[3*tid+2];
	// C = (Public_A - Delta_A) * (Public_B - Delta_B)
	int64_t res_local = (party - 1)*Public_A*Public_B - Public_A*Delta_B - Delta_A*Public_B + Delta_C;
	res[tid] = res_local;
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

__global__ void fss_pack_gen_kernel(uint32_t key[4 * (14 + 1)], uint64_t* alpha, int n, DCF_Keys k0, DCF_Keys k1, int N, int maxlayer, uint128_t* g_prefix_ones_table) {

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
	dcf_pack_gen_device(&prg, expanded_key, alpha_tid, n, maxlayer, k0_local, k1_local, g_prefix_ones_table);

}

__global__ void fss_pack_eval_kernel(bool* res, uint32_t key[4 * (14 + 1)], uint64_t* alpha, DCF_Keys k, int N, int maxlayer) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	unsigned char* k_local = (unsigned char*) (k + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t alpha_tid(0, alpha[tid]);
	res[tid] = dcf_pack_eval_device(expanded_key, k_local, alpha_tid, maxlayer).get_lsb();

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

__global__ void circular_shift_kernel(int64_t* output, int64_t* input, int shift) {
    // 线程索引
    int tid = threadIdx.x;
    
    // 方法1：使用共享内存
    __shared__ int64_t shared_data[256];
    
    // 存入共享内存
    shared_data[tid] = input[tid];
    
    // 同步所有线程
    __syncthreads();
    
    // 计算目标位置（循环右移一位）
    int target_pos = (tid + shift) > 0 ? (tid + shift) % 256 : 256 + (tid + shift);
    
    // 从共享内存读取移位后的值
    int64_t shifted_value = shared_data[target_pos];
    
    // 存储结果
    output[tid] = shifted_value;
}




// 初始化kernel
__global__ void init_prefix_ones_table(uint128_t* g_prefix_ones_table) {
    int idx = threadIdx.x;
    if (idx >= 128) return;
    
    if (idx < 64) {
        // 前64项：只设置高64位
        uint64_t high = (~0ULL << (63 - idx));
        g_prefix_ones_table[idx] = uint128_t(high, 0);
    } else {
        // 后64项：高64位全1，设置低64位
        uint64_t low = (~0ULL << (127 - idx));
        g_prefix_ones_table[idx] = uint128_t(0xFFFFFFFFFFFFFFFFULL, low);
    }
}

__global__ void init_selection_table(uint128_t* g_selection_table) {
    int idx = threadIdx.x;
    if (idx >= 128) return;
    
    if (idx < 64) {
        // 前64项：只设置高64位
        uint64_t high = (1ULL << (63 - idx));
        g_selection_table[idx] = uint128_t(high, 0);
    } else {
        // 后64项：高64位全1，设置低64位
        uint64_t low = (1ULL << (127 - idx));
        g_selection_table[idx] = uint128_t(0, low);
    }
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

	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	//move MSB_keys to device
	// cudaEventRecord(start);
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
	// cudaEventRecord(stop);

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	int select = party == 1 ? 1 : 0;

	// cudaEvent_t start1, stop1;
    // cudaEventCreate(&start1);
    // cudaEventCreate(&stop1);

	// cudaEventRecord(start1);
	fss_msb_eval_kernel<<<blocks, threads>>>(res_device, aes_key, k_device, value_device, r_msb_device, N, maxlayer, select);


	cudaMemcpy(res, res_device, N * sizeof(bool), cudaMemcpyDeviceToHost);
	// 等待GPU完成
    cudaDeviceSynchronize();

	// cudaEventRecord(stop1);
	// cudaEventSynchronize(stop1);
	// float time1 = 0;
    // cudaEventElapsedTime(&time1, start1, stop1);
    // printf("msb eval Kernel Time taken: %.3f ms\n", time1);


	
	// float time = 0;
	// cudaEventElapsedTime(&time, start, stop);
	// printf("msb memcpy Time taken: %.3f ms\n", time);
	cudaFree(k_device);
	cudaFree(value_device);
	cudaFree(r_msb_device);
	cudaFree(res_device);
	cudaFree(aes_key);
}

extern "C" void cudamsbeval_buf(bool* res, DCF_Keys k_device, int64_t* value, bool* r_msb_device, int N, int maxlayer, int party){

	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	//move MSB_keys to device
	// cudaEventRecord(start);
	// DCF_Keys k_device;
	// k_device = keys.k;
	// cudaMalloc(&k_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	// cudaMemcpy(k_device, k, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	int64_t* value_device;
	cudaMalloc(&value_device, N * sizeof(int64_t));
	cudaMemcpy(value_device, value, N * sizeof(int64_t), cudaMemcpyHostToDevice);

	// bool* r_msb_device;
	// r_msb_device = keys.r_msb;
	// cudaMalloc(&r_msb_device, N * sizeof(bool));
	// cudaMemcpy(r_msb_device, r_msb, N * sizeof(bool), cudaMemcpyHostToDevice);

	bool* res_device;
	cudaMalloc(&res_device, N * sizeof(bool));
	// cudaEventRecord(stop);

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	int select = party == 1 ? 1 : 0;

	// cudaEvent_t start1, stop1;
    // cudaEventCreate(&start1);
    // cudaEventCreate(&stop1);

	// cudaEventRecord(start1);
	fss_msb_eval_kernel<<<blocks, threads>>>(res_device, aes_key, k_device, value_device, r_msb_device, N, maxlayer, select);


	cudaMemcpy(res, res_device, N * sizeof(bool), cudaMemcpyDeviceToHost);
	// 等待GPU完成
    cudaDeviceSynchronize();

	// cudaEventRecord(stop1);
	// cudaEventSynchronize(stop1);
	// float time1 = 0;
    // cudaEventElapsedTime(&time1, start1, stop1);
    // printf("msb eval Kernel Time taken: %.3f ms\n", time1);


	
	// float time = 0;
	// cudaEventElapsedTime(&time, start, stop);
	// printf("msb memcpy Time taken: %.3f ms\n", time);
	cudaFree(value_device);
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

	uint128_t* g_prefix_ones_table;
	cudaMalloc(&g_prefix_ones_table, 128 * sizeof(uint128_t));
	init_prefix_ones_table<<<1, 128>>>(g_prefix_ones_table);
	cudaDeviceSynchronize();

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	fss_pack_gen_kernel<<<blocks, threads>>>(aes_key, alpha_device, n, k0_device, k1_device, N, maxlayer, g_prefix_ones_table);

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
    fss_pack_eval_kernel<<<blocks, threads>>>(res_device, aes_key, value_device, key_device, N, maxlayer);   

	cudaMemcpy(res, res_device, N * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaFree(key_device);
	cudaFree(value_device);
}

extern "C" void cudalutkeygen(DCF_Keys k0, DCF_Keys k1, uint64_t* alpha, int N, int n,int maxlayer){

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
	lut_gen_kernel<<<blocks, threads>>>(aes_key, alpha_device, n, k0_device, k1_device, N, maxlayer);

	cudaMemcpy(k0, k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(k1, k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);	

	cudaFree(k0_device);
	cudaFree(k1_device);
	cudaFree(alpha_device);
	cudaFree(aes_key);
}

extern "C" void cudaluteval(int64_t *res, DCF_Keys key, uint64_t *value,int N, int maxlayer, int party){

	DCF_Keys key_device;
	cudaMalloc(&key_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	cudaMemcpy(key_device, key, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	uint64_t* value_device;
	cudaMalloc(&value_device, N * sizeof(uint64_t));
	cudaMemcpy(value_device, value, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

	int64_t* res_device;
	cudaMalloc(&res_device, N * sizeof(int64_t));

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;	
    lut_eval_kernel<<<blocks, threads>>>(res_device, aes_key, value_device, key_device, N, maxlayer, party);   

	cudaMemcpy(res, res_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);

	cudaFree(key_device);
	cudaFree(value_device);
}

extern "C" void cudalutevalall(int64_t *res, DCF_Keys key,int* shift, int N, int maxlayer, int party){

	DCF_Keys key_device;
	cudaMalloc(&key_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	cudaMemcpy(key_device, key, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	int64_t* res_device;
	cudaMalloc(&res_device, N * 256 * sizeof(int64_t));

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int* shift_device;
	cudaMalloc(&shift_device, N * sizeof(int));
	cudaMemcpy(shift_device, shift, N * sizeof(int), cudaMemcpyHostToDevice);
	// int threads = N > 256 ? 256 : N;
	// int blocks = (N + threads - 1) / threads;	
	cudaEvent_t start, stop;	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	lut_eval_all_kernel<<<N, 128>>>(res_device, aes_key, key_device, shift_device, N, maxlayer, party);   

	cudaMemcpy(res, res_device, N * 256 * sizeof(int64_t), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	printf("Eval CUDA Time taken: %f milliseconds\n", time);

	cudaFree(key_device);
	cudaFree(res_device);
	cudaFree(shift_device);
}

extern "C" void cudalutPackkeygen(DCF_Keys k0, DCF_Keys k1, uint64_t* alpha, int N){

	int n = 8;
	int maxlayer = 1;

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

	uint128_t* g_selection_table;
	cudaMalloc(&g_selection_table, 128 * sizeof(uint128_t));
	init_selection_table<<<1, 128>>>(g_selection_table);
	cudaDeviceSynchronize();

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	lut_pack_gen_kernel<<<blocks, threads>>>(aes_key, alpha_device, n, k0_device, k1_device, N, maxlayer, g_selection_table);

	cudaMemcpy(k0, k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(k1, k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);	

	cudaFree(k0_device);
	cudaFree(k1_device);
	cudaFree(alpha_device);
	cudaFree(aes_key);
}

extern "C" void cudalutPackevalall(int64_t *res, DCF_Keys key,int* shift, int N, int party){

	int maxlayer = 1;
	DCF_Keys key_device;
	cudaMalloc(&key_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	cudaMemcpy(key_device, key, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	int64_t* res_device;
	cudaMalloc(&res_device, N  * sizeof(int64_t));

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int* shift_device;
	cudaMalloc(&shift_device, N * sizeof(int));
	cudaMemcpy(shift_device, shift, N * sizeof(int), cudaMemcpyHostToDevice);
	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;	
	// cudaEvent_t start, stop;	
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	// cudaEventRecord(start);
	lut_pack_eval_kernel<<<blocks, threads>>>(res_device, aes_key, key_device, shift_device, N, maxlayer);   

	cudaMemcpy(res, res_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);

	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// float time = 0;
	// cudaEventElapsedTime(&time, start, stop);
	// printf("Eval CUDA Time taken: %f milliseconds\n", time);

	cudaFree(key_device);
	cudaFree(res_device);
	cudaFree(shift_device);
}

extern "C" void LUTkeygen(DCF_Keys k0, DCF_Keys k1, 
						int64_t* random_in0, int64_t* random_in1, 
						uint64_t* random_out0, uint64_t* random_out1, 
						int64_t* random_out0_A, int64_t* random_out1_A, 
						int N){

	int n = 8;
	int maxlayer = 1;

	DCF_Keys k0_device;
	cudaMalloc(&k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	DCF_Keys k1_device;
	cudaMalloc(&k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	int64_t* random_in0_device;
	cudaMalloc(&random_in0_device, N * sizeof(int64_t));

	int64_t* random_in1_device;
	cudaMalloc(&random_in1_device, N * sizeof(int64_t));

	uint64_t* random_out0_device;
	cudaMalloc(&random_out0_device, N * sizeof(uint64_t));

	uint64_t* random_out1_device;
	cudaMalloc(&random_out1_device, N * sizeof(uint64_t));

	int64_t* random_out0_A_device;
	cudaMalloc(&random_out0_A_device, 64 * N * sizeof(int64_t));

	int64_t* random_out1_A_device;
	cudaMalloc(&random_out1_A_device, 64 * N * sizeof(int64_t));

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	uint128_t* g_selection_table;
	cudaMalloc(&g_selection_table, 128 * sizeof(uint128_t));
	init_selection_table<<<1, 128>>>(g_selection_table);
	cudaDeviceSynchronize();

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	fss_lut_keygen_kernel<<<blocks, threads>>>(aes_key,k0_device, k1_device, random_in0_device, random_in1_device, random_out0_device, random_out1_device, random_out0_A_device, random_out1_A_device, N, n, maxlayer, g_selection_table);

	cudaMemcpy(k0, k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(k1, k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);	
	cudaMemcpy(random_in0, random_in0_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(random_in1, random_in1_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(random_out0, random_out0_device, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(random_out1, random_out1_device, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(random_out0_A, random_out0_A_device, 64 * N * sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(random_out1_A, random_out1_A_device, 64 * N * sizeof(int64_t), cudaMemcpyDeviceToHost);

	cudaFree(k0_device);
	cudaFree(k1_device);
	cudaFree(random_in0_device);
	cudaFree(random_in1_device);
	cudaFree(random_out0_device);
	cudaFree(random_out1_device);
	cudaFree(random_out0_A_device);
	cudaFree(random_out1_A_device);
	cudaFree(aes_key);
}

extern "C" void LUTeval(int64_t *res_device, DCF_Keys keys_device, uint64_t *random_out_device, int64_t *value_device, int N, int maxlayer, int party){

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;	
	fss_lut_eval_kernel<<<blocks, threads>>>(res_device, aes_key, keys_device, random_out_device, value_device, N, maxlayer);   
	cudaDeviceSynchronize();
}

extern "C" void LUTlinear(int64_t* res_shared, int64_t* open_res, int64_t* random_out, int N, int party){

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;	
	fss_lut_linear_kernel<<<blocks, threads>>>(res_shared, open_res, random_out, N, party);
	cudaDeviceSynchronize();
}

extern "C" void LUTSelectionKeygen(uint128_t* k0_device, uint128_t* k1_device, 
						int64_t* random_in0_device, int64_t* random_in1_device, 
						int64_t* random_out0_device, int64_t* random_out1_device, 
						int N){

	uint32_t* aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	uint128_t* g_selection_table;
	cudaMalloc(&g_selection_table, 128 * sizeof(uint128_t));
	init_selection_table<<<1, 128>>>(g_selection_table);
	cudaDeviceSynchronize();

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	fss_lutSelection_keygen_kernel<<<blocks, threads>>>(aes_key, k0_device, k1_device, random_in0_device, random_in1_device, random_out0_device, random_out1_device, N, g_selection_table);
	cudaDeviceSynchronize();


}

extern "C" void LUTSelectionEval(int64_t* res_device, uint128_t* k_device, int64_t* random_out, int64_t* value, int N, int party){

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;	
	fss_lutSelection_eval_kernel<<<blocks, threads>>>(res_device, k_device, random_out, value, N, party);
	cudaDeviceSynchronize();
}

extern "C" void LUTSelectionLinear(int64_t* res_shared, int64_t* Public_value, int64_t* random_out, int N, int party){

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;	
	fss_lutSelection_linear_kernel<<<blocks, threads>>>(res_shared, Public_value, random_out, N, party);
	cudaDeviceSynchronize();
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

// 主机端代码
extern "C" void test_circular_shift(int64_t* output_h, int64_t* input_h, int shift) {
    // 分配设备内存
	int64_t* input_d;
	cudaMalloc(&input_d, 256 * sizeof(int64_t));
	cudaMemcpy(input_d, input_h, 256 * sizeof(int64_t), cudaMemcpyHostToDevice);
    int64_t *d_output;
    cudaMalloc(&d_output, 256 * sizeof(int64_t));

    // 启动核函数
	circular_shift_kernel<<<1, 256>>>(d_output, input_d, shift);

    // 分配主机内存并复制结果
    cudaMemcpy(output_h, d_output, 256 * sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    // 清理内存
    cudaFree(d_output);
}