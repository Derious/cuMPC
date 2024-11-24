#ifndef UINT128_TYPE_H
#define UINT128_TYPE_H

#include <cstdint>
#include <stdio.h>
#include <cuda_runtime.h>



class uint128_t {
private:
    struct alignas(16) {
        uint64_t high;
        uint64_t low;
    };

public:
    // constructor
    __host__ __device__ uint128_t() : high(0), low(0) {}
    __host__ __device__ uint128_t(uint64_t h, uint64_t l) : high(h), low(l) {}
    
    // get high and low
    __host__ __device__ uint64_t get_high() const { return high; }
    __host__ __device__ uint64_t get_low() const { return low; }

    //set Last bit zero
    __host__ __device__ uint128_t set_lsb_zero() {
        low = low & 0xFFFFFFFFFFFFFFFE;
        return *this;
    }

    // reverse lsb
    __host__ __device__ uint128_t reverse_lsb() {
        low = low ^ 0x1;
        return *this;
    }

    // get lsb
    __host__ __device__ int get_lsb() {
        return low & 1;
    }
    
    // bitwise operator
    __host__ __device__ uint128_t operator^(const uint128_t& other) const {
        return uint128_t(high ^ other.high, low ^ other.low);
    }
    
    __host__ __device__ uint128_t operator&(const uint128_t& other) const {
        return uint128_t(high & other.high, low & other.low);
    }
    
    __host__ __device__ uint128_t operator|(const uint128_t& other) const {
        return uint128_t(high | other.high, low | other.low);
    }

    // add operator
    __host__ __device__ uint128_t operator+(const uint128_t& other) const {
        uint64_t new_high = high + other.high;
        uint64_t new_low = low + other.low;
        if (new_low < low) {
            new_high++;
        }
        return uint128_t(new_high, new_low);
    }

    // add assign operator
    __host__ __device__ uint128_t& operator+=(const uint128_t& other) {
        *this = *this + other;
        return *this;
    }

    // select operator
    __host__ __device__ uint128_t select(int bit) const {
        return uint128_t(high * bit, low * bit);
    }

    // optimized left shift operator
    __host__ __device__ __forceinline__ uint128_t operator<<(unsigned int shift) const {
        // use bit mask to optimize branch
        const uint64_t mask = shift >= 64 ? 0xFFFFFFFFFFFFFFFF : 0;
        const unsigned int effective_shift = shift & 63;  // shift % 64
        
        uint64_t new_high, new_low;
        
        // use conditional selection instead of branch
        new_high = (~mask & ((high << effective_shift) | 
                   ((low >> (64 - effective_shift)) & ~(-1LL << effective_shift))));
        new_low = (~mask & (low << effective_shift)) | 
                  ((mask & high) << effective_shift);
        
        return uint128_t(new_high, new_low);
    }

    // optimized right shift operator
    __host__ __device__ __forceinline__ uint128_t operator>>(unsigned int shift) const {
        const uint64_t mask = shift >= 64 ? 0xFFFFFFFFFFFFFFFF : 0;
        const unsigned int effective_shift = shift & 63;  // shift % 64
        
        uint64_t new_high, new_low;
        
        new_high = (~mask & (high >> effective_shift));
        new_low = ((~mask & (low >> effective_shift)) | 
                  (~mask & (high << (64 - effective_shift)))) |
                  ((mask & high) >> effective_shift);
        
        return uint128_t(new_high, new_low);
    }

    // left shift assign operator
    __host__ __device__ uint128_t& operator<<=(unsigned int shift) {
        *this = *this << shift;
        return *this;
    }

    // right shift assign operator
    __host__ __device__ uint128_t& operator>>=(unsigned int shift) {
        *this = *this >> shift;
        return *this;
    }

    // get bit
    __host__ __device__ bool get_bit(unsigned int index) const {
        return (*this >> index).low & 1;
    }

    // byte conversion
    __host__ __device__ void to_bytes(uint8_t* bytes) const {
        for(int i = 0; i < 8; i++) {
            bytes[i] = (high >> (56 - i * 8)) & 0xFF;
            bytes[i + 8] = (low >> (56 - i * 8)) & 0xFF;
        }
    }
    
    __host__ __device__ static uint128_t from_bytes(const uint8_t* bytes) {
        uint64_t h = 0, l = 0;
        for(int i = 0; i < 8; i++) {
            h = (h << 8) | bytes[i];
            l = (l << 8) | bytes[i + 8];
        }
        return uint128_t(h, l);
    }

    // return a pointer to the bytes of uint128_t
    __host__ __device__ uint8_t* get_bytes() {
        return reinterpret_cast<uint8_t*>(&high);
    }
    
    // compare operator
    __host__ __device__ bool operator==(const uint128_t& other) const {
        return high == other.high && low == other.low;
    }
    
    __host__ __device__ bool operator!=(const uint128_t& other) const {
        return !(*this == other);
    }

    // print uint128
    __host__ __device__ void print_uint128(const char* prefix, const uint128_t& val) {
        printf("%s%016lx%016lx\n", prefix, val.get_high(), val.get_low());
    }
};

#endif // UINT128_TYPE_H