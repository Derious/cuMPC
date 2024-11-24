#include <iostream>
#include <iomanip>

void generateSelectVectorTable() {
    std::cout << "const uint128_t select_vector_table[128] = {\n";
    
    // 生成高64位部分 (索引 0-63)
    for(int i = 0; i < 64; i++) {
        uint64_t high = 1ULL << (63 - i);
        uint64_t low = 0ULL;
        
        std::cout << "    uint128_t(0x" 
                  << std::hex << std::setfill('0') << std::setw(16) << high 
                  << "ULL, 0x"
                  << std::hex << std::setfill('0') << std::setw(16) << low 
                  << "ULL),";
        
        std::cout << " // " << std::dec << i << "\n";
    }
    
    // 生成低64位部分 (索引 64-127)
    for(int i = 64; i < 128; i++) {
        uint64_t high = 0ULL;
        uint64_t low = 1ULL << (127 - i);
        
        std::cout << "    uint128_t(0x" 
                  << std::hex << std::setfill('0') << std::setw(16) << high 
                  << "ULL, 0x"
                  << std::hex << std::setfill('0') << std::setw(16) << low 
                  << "ULL)";
        
        // 最后一个元素不需要逗号
        if (i < 127) std::cout << ",";
        std::cout << " // " << std::dec << i << "\n";
    }
    
    std::cout << "};\n";
}

int main() {
    generateSelectVectorTable();
    return 0;
}