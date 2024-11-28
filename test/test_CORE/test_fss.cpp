#include "../../mpc_cuda/mpc_core.h"
#include <iostream>
#include <vector>
#include <chrono>

// 测试代码
int main() {

    uint64_t userkey1 = 597349; uint64_t userkey2 = 121379; 
	block userkey = makeBlock(userkey1, userkey2);
    uint64_t plaintext1 = 597349; uint64_t plaintext2 = 121379; 
	uint128_t plaintext(plaintext1, plaintext2);
    uint128_t ciphertext;
    AES_KEY key_host;
	AES_set_encrypt_key(userkey, &key_host);
    // AES_ecb_encrypt(plaintext.get_bytes(), ciphertext.get_bytes(), &key_host,1);


    uint128_t ciphertext_2;
    int bit1, bit2;
    Double_PRG(&key_host, plaintext, ciphertext, ciphertext_2, bit1, bit2);
    plaintext.print_uint128("plaintext = ", plaintext);
    ciphertext.print_uint128("ciphertext1 = ", ciphertext);
    ciphertext_2.print_uint128("ciphertext2 = ", ciphertext_2);
    printf("bit1 = %d, bit2 = %d\n", bit1, bit2);

    AES_Generator prg;
    uint128_t output1, output2;
    uint8_t* k0, *k1;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 30520; i++){
        fss_gen(&prg, &key_host, uint128_t(0, 5), 64, &k0, &k1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Keygen Time taken: %f milliseconds\n", elapsed.count() * 1000 / 30520);

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 30520; i++){
        output1 = dcf_eval(&key_host, k0, uint128_t(0, i));
        // output2 = dcf_eval(&key_host, k1, uint128_t(0, i));
        // printf("output1 = %lu, output2 = %lu\n", output1.get_low(), output2.get_low());
        // uint128_t res = output1 ^ output2;
        // res.print_uint128("res = ", res);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("CPU Time taken: %f milliseconds\n", elapsed.count() * 1000 / 30520);

    int maxlayer = 32-7;
    start = std::chrono::high_resolution_clock::now();

    uint64_t random_num = (prg.random().get_low()) % 0xFFFFFFFF;
    // uint64_t random_num = 0x8bfbfb1c76bc43cf;
    Dcf_Pack(&prg, &key_host, uint128_t(0, random_num), 32, &k0, &k1, maxlayer);
    printf("random_num = %lx\n", random_num);
    for(uint32_t i = 0; i <= 0xFFFF; i++){    
        bool res1 = Dcf_Pack_Eval(&key_host, k0, uint128_t(0, i), maxlayer);
        bool res2 = Dcf_Pack_Eval(&key_host, k1, uint128_t(0, i), maxlayer);
        int res_final = res1 ^ res2;
        if(res_final != (i<=random_num)){
            printf("random_num = %lx, i = %x\n", random_num, i);
            printf("res = %d\n", res_final);
        }

        // uint128_t res11 = EVAL_Pack(&key_host, k0, uint128_t(0, i));
        // uint128_t res22 = EVAL_Pack(&key_host, k1, uint128_t(0, i));
        // uint128_t res_final2 = res11 ^ res22;
        // res_final2.print_uint128("res_final2 = ", res_final2);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("Eval Time taken: %f milliseconds\n", elapsed.count() * 1000 / 0xFFFF);


    printf("===============================================\n");

    cudaWarmup(512, 0);
    int N = 30520;
    int n = 64;
    maxlayer = n - 7;
    DCF_Keys dcf_k0, dcf_k1;
    dcf_k0 = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
    dcf_k1 = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
    uint64_t* alpha = new uint64_t[N];
    uint64_t* alpha2 = new uint64_t[N];
    for(int i = 0; i < N; i++){
        alpha[i] = 0x100;
        alpha2[i] = i;
    }

    start = std::chrono::high_resolution_clock::now();
    cudafsskeygen(dcf_k0, dcf_k1, alpha, N, n, maxlayer);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("Keygen CUDA Time taken: %f milliseconds\n", elapsed.count() * 1000);

    bool* res1 = new bool[N];
    bool* res2 = new bool[N];
    start = std::chrono::high_resolution_clock::now();
    cudafsseval(res1, dcf_k0, alpha2, N, maxlayer, 0);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("Eval CUDA Time taken: %f milliseconds\n", elapsed.count() * 1000);

    cudafsseval(res2, dcf_k1, alpha2, N, maxlayer, 1);

    start = std::chrono::high_resolution_clock::now();
    uint32_t final_res = 0;
    for(int i = 0; i < N; i++){
        final_res += (uint32_t)(res1[i]^res2[i]);
        if( res1[i]^res2[i] ^ (i<=0x100)){
            printf("Error at i = %d, res1[i]^res2[i] = %d, i<=0x100 = %d\n", i, res1[i]^res2[i], i<=0x100);
        }
    }
    printf("final_res = %d\n", final_res);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("Final res reduction Time taken: %f milliseconds\n", elapsed.count() * 1000);

    free(dcf_k0);
    free(dcf_k1);
    delete[] alpha;
    delete[] res1;
    delete[] res2;

    return 0;
}