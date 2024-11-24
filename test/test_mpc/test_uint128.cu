#include "../../mpc_keys/uint128_type.h"
#include <cassert>


// 主机端测试函数
void test_host_uint128() {
    printf("\n=== Testing uint128_t on Host ===\n");
    
    // 构造函数测试
    uint128_t a(0x1234567890ABCDEF, 0xFEDCBA9876543210);
    uint128_t b(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    uint128_t zero;
    
    a.print_uint128("a = ", a);
    b.print_uint128("b = ", b);
    zero.print_uint128("zero = ", zero);

    // 位运算测试
    uint128_t xor_result = a ^ b;
    uint128_t and_result = a & b;
    uint128_t or_result = a | b;
    
    printf("\nBitwise Operations:\n");
    xor_result.print_uint128("a ^ b = ", xor_result);
    and_result.print_uint128("a & b = ", and_result);
    or_result.print_uint128("a | b = ", or_result);

    // 字节转换测试
    uint8_t bytes[16];
    a.to_bytes(bytes);
    uint128_t c = uint128_t::from_bytes(bytes);
    printf("\nByte conversion test:\n");
    a.print_uint128("Original: ", a);
    c.print_uint128("After conversion: ", c);
    assert(a.get_high() == c.get_high() && a.get_low() == c.get_low() && "Byte conversion failed!");
}

// CUDA设备端测试函数
__global__ void test_device_uint128() {
    printf("\n=== Testing uint128_t on Device ===\n");
    
    // 构造函数测试
    uint128_t a(0x1234567890ABCDEF, 0xFEDCBA9876543210);
    uint128_t b(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    uint128_t zero;
    
    a.print_uint128("a = ", a);
    b.print_uint128("b = ", b);
    zero.print_uint128("zero = ", zero);

    // 位运算测试
    uint128_t xor_result = a ^ b;
    uint128_t and_result = a & b;
    uint128_t or_result = a | b;
    
    printf("\nBitwise Operations:\n");
    xor_result.print_uint128("a ^ b = ", xor_result);
    and_result.print_uint128("a & b = ", and_result);
    or_result.print_uint128("a | b = ", or_result);

    // 字节转换测试
    uint8_t bytes[16];
    a.to_bytes(bytes);
    uint128_t c = uint128_t::from_bytes(bytes);
    printf("\nByte conversion test:\n");
    a.print_uint128("Original: ", a);
    c.print_uint128("After conversion: ", c);
    assert(a.get_high() == c.get_high() && a.get_low() == c.get_low() && "Byte conversion failed!");
}

// 验证函数
bool verify_results(const uint128_t& val, uint64_t expected_high, uint64_t expected_low) {
    return val.get_high() == expected_high && val.get_low() == expected_low;
}

int main() {
    // 运行主机端测试
    test_host_uint128();

    // 运行设备端测试
    test_device_uint128<<<1,1>>>();
    cudaDeviceSynchronize();

    // 额外的验证测试
    printf("\n=== Running Additional Verification Tests ===\n");
    
    // 测试基本运算
    uint128_t test1(1, 0);
    uint128_t test2(0, 1);
    uint128_t result = test1 ^ test2;
    
    assert(verify_results(result, 1, 1) && "XOR operation failed!");
    printf("XOR test passed\n");

    // 测试边界情况
    uint128_t max_val(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    uint128_t zero_val;
    result = max_val & zero_val;
    
    assert(verify_results(result, 0, 0) && "AND operation with zero failed!");
    printf("Zero AND test passed\n");

    // 测试复杂组合
    uint128_t val1(0x1234567890ABCDEF, 0xFEDCBA9876543210);
    uint128_t val2(0xFFFFFFFFFFFFFFFF, 0);
    result = (val1 & val2) | test1;
    
    assert(verify_results(result, 0x1234567890ABCDEF, 1) && "Complex operation failed!");
    printf("Complex operation test passed\n");

    printf("\nAll tests completed successfully!\n");
    
    return 0;
} 