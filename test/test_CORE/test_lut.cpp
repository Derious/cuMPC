#include <fstream>
#include <iomanip>
#include <iostream>

struct uint128_t {
    unsigned long long high;
    unsigned long long low;
    
    uint128_t() : high(0), low(0) {}
    uint128_t(unsigned long long h, unsigned long long l) : high(h), low(l) {}
};

void generate_prefix_ones_table() {
    std::ofstream outFile("prefix_ones_table.txt");
    
    outFile << "// prefix ones table from 0 to 127 (starting from highest bit)\n";
    outFile << "extern const uint128_t prefix_ones_table[128] = {\n";
    
    // 处理前64位（high部分）
    for (int i = 0; i < 64; i++) {
        unsigned long long high = (~0ULL << (63 - i));  // 从最高位开始填1
        unsigned long long low = 0;
        outFile << "    uint128_t(0x" << std::hex << std::setfill('0') << std::setw(16) 
               << high << "ULL, 0x" << std::setfill('0') << std::setw(16) 
               << low << "ULL), // " 
               << std::dec << i << "\n";
    }
    
    // 处理后64位（low部分）
    for (int i = 64; i < 128; i++) {
        unsigned long long high = ~0ULL;  // 高64位全是1
        unsigned long long low = (~0ULL << (127 - i));  // 低64位从高位开始填1
        outFile << "    uint128_t(0x" << std::hex << std::setfill('0') << std::setw(16) 
               << high << "ULL, 0x" << std::setfill('0') << std::setw(16) 
               << low << "ULL), // " 
               << std::dec << i << "\n";
    }
    
    outFile << "};\n";
    outFile.close();
}

int main() {
    generate_prefix_ones_table();
    std::cout << "Prefix ones table has been generated to prefix_ones_table.txt\n";
    return 0;
}