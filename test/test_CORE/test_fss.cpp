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

    auto start = std::chrono::high_resolution_clock::now();
    uint8_t* k0, *k1;
    fss_gen(&prg, &key_host, uint128_t(0, 5), 64, &k0, &k1);
    for(int i = 0; i < 10000; i++){
        output1 = dcf_eval(&key_host, k0, uint128_t(0, i));
        output2 = dcf_eval(&key_host, k1, uint128_t(0, i));
        printf("output1 = %lu, output2 = %lu\n", output1.get_low(), output2.get_low());
        uint128_t res = output1 ^ output2;
        res.print_uint128("res = ", res);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("CPU Time taken: %f milliseconds\n", elapsed.count() * 1000);

    printf("===============================================\n");

    cudaWarmup(512, 0);
    int N = 10000;
    int maxlayer = 64;
    DCF_Keys dcf_k0, dcf_k1;
    dcf_k0 = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
    dcf_k1 = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
    uint64_t* alpha = new uint64_t[N];
    uint64_t* alpha2 = new uint64_t[N];
    for(int i = 0; i < N; i++){
        alpha[i] = 64;
        alpha2[i] = i;
    }

    start = std::chrono::high_resolution_clock::now();
    cudafsskeygen(dcf_k0, dcf_k1, alpha, N, 64, maxlayer);
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

    // for(int i = 0; i < N; i++){
    //     printf("res1[%d]^res2[%d] = %d\n", i, i, res1[i]^res2[i]);
    // }

    free(dcf_k0);
    free(dcf_k1);
    delete[] alpha;
    delete[] res1;
    delete[] res2;

    return 0;
}