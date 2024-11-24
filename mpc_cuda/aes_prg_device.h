#ifndef AES_PRG_HOST_H
#define AES_PRG_HOST_H

#include "../mpc_keys/uint128_type.h"
#include "aes_cuda.h"
#include <curand.h>
#include <curand_kernel.h>

class AES_Generator_device {
private:
    uint32_t key[4 * (14 + 1)];
    uint128_t counter;  // 用作计数器

    
    // 初始化计数器 (CUDA版本)
    __device__ static uint128_t init_counter() {
        // 使用curand生成两个64位随机数
        int idx = threadIdx.x;
        curandState state;
        curand_init(clock64(), idx, 0, &state);  // 初始化种子
        uint64_t high = (static_cast<uint64_t>(curand(&state)) << 32) | curand(&state);
        uint64_t low = (static_cast<uint64_t>(curand(&state)) << 32) | curand(&state);
        return uint128_t(high, low);
    }

public:
    // 构造函数需要传入curandState
    __device__ AES_Generator_device() : counter(init_counter()) {
        uint64_t userkey1 = 597349; uint64_t userkey2 = 121379; 
        uint128_t userkey(userkey1, userkey2);
        if (AES_set_encrypt_key_cu(userkey.get_bytes(), 128, key) != 0) {
        printf("Key expansion failed!\n");
        return;
    }
    }

    // 生成随机数
    __device__ uint128_t random() {
        uint128_t result;
        result = counter;
        AES_encrypt_cu(result.get_bytes(), result.get_bytes(), key);
        counter += uint128_t(1, 0);  // 增加计数器
        return result;
    }
};

#endif // AES_PRG_HOST_H