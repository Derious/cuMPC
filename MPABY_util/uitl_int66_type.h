#ifndef __INT66TYPE
#define __INT66TYPE

#include <iostream>
#include <cstdint>

#include <iostream>
#include <cstdint>

struct int66_t {
    uint64_t low;  // 低64位
    uint8_t high;  // 高2位

    // 构造函数
    int66_t() : low(0), high(0) {}

    // 使用低位初始化
    int66_t(uint64_t l) : low(l), high(0) {}

    // 使用高位和低位初始化
    int66_t(uint64_t l, uint8_t h) : low(l), high(h & 0x3) {}  // 确保高位只占2位

    // 打印函数
    void print() const {
        std::cout << "High: " << static_cast<uint16_t>(high) << ", Low: " << low << std::endl;
    }
};

// 加法运算符
int66_t operator+(const int66_t &a, const int66_t &b) {
    int66_t result;
    uint64_t low_sum = a.low + b.low;
    result.low = low_sum;
    uint8_t carry = (low_sum < a.low) ? 1 : 0;
    result.high = (a.high + b.high + carry) & 0x3;
    return result;
}

// 减法运算符
int66_t operator-(const int66_t &a, const int66_t &b) {
    int66_t result;
    uint64_t low_sub = a.low - b.low;
    result.low = low_sub;
    uint8_t borrow = (a.low < b.low) ? 1 : 0;
    result.high = (a.high - b.high - borrow) & 0x3;
    return result;
}

// 乘法运算符
int66_t operator*(const int66_t &a, const int66_t &b) {
    int66_t result;
    __uint128_t product = static_cast<__uint128_t>(a.low) * static_cast<__uint128_t>(b.low);
    result.low = static_cast<uint64_t>(product);
    uint64_t high_part = static_cast<uint64_t>(product >> 64);
    result.high = (a.high * b.low + b.high * a.low + high_part) & 0x3;
    return result;
}

// // 除法运算符
// int66_t operator/(const int66_t &a, const int66_t &b) {
//     if (b.low == 0 && b.high == 0) throw std::overflow_error("Division by zero");
//     __uint128_t dividend = (static_cast<__uint128_t>(a.high) << 64) | a.low;
//     __uint128_t divisor = (static_cast<__uint128_t>(b.high) << 64) | b.low;
//     __uint128_t quotient = dividend / divisor;
//     return int66_t(static_cast<uint64_t>(quotient), 0);
// }

// // 取模运算符
// int66_t operator%(const int66_t &a, const int66_t &b) {
//     if (b.low == 0 && b.high == 0) throw std::overflow_error("Modulo by zero");
//     __uint128_t dividend = (static_cast<__uint128_t>(a.high) << 64) | a.low;
//     __uint128_t divisor = (static_cast<__uint128_t>(b.high) << 64) | b.low;
//     __uint128_t remainder = dividend % divisor;
//     return int66_t(static_cast<uint64_t>(remainder), 0);
// }

// 比较运算符
bool operator==(const int66_t &a, const int66_t &b) {
    return a.low == b.low && a.high == b.high;
}

bool operator!=(const int66_t &a, const int66_t &b) {
    return !(a == b);
}

bool operator<(const int66_t &a, const int66_t &b) {
    if (a.high != b.high)
        return a.high < b.high;
    return a.low < b.low;
}

bool operator>(const int66_t &a, const int66_t &b) {
    return b < a;
}

bool operator<=(const int66_t &a, const int66_t &b) {
    return !(a > b);
}

bool operator>=(const int66_t &a, const int66_t &b) {
    return !(a < b);
}

#endif