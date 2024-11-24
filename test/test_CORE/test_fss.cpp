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
    uint64_t random = prg.random().get_low();
    uint64_t random2 = prg.random().get_low();
    uint128_t output1, output2;

    uint8_t* k0, *k1;
    fss_gen(&prg, &key_host, uint128_t(0, random), 64, &k0, &k1);

    printf("===============================================\n");

    auto start = std::chrono::high_resolution_clock::now();
    output1 = dcf_eval(&key_host, k0, uint128_t(0, random2));
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Time taken: %f milliseconds\n", elapsed.count() * 1000);
    output2 = dcf_eval(&key_host, k1, uint128_t(0, random2));
    output1.print_uint128("output1 = ", output1);
    output2.print_uint128("output2 = ", output2);
    uint128_t res = output1 ^ output2;
    res.print_uint128("res = ", res);
    printf("random < random2 = %d\n", (random < random2));
    

    cudaWarmup(256, 0);
    start = std::chrono::high_resolution_clock::now();
    test_dcf();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("CUDA Time taken: %f milliseconds\n", elapsed.count() * 1000);

    return 0;
}