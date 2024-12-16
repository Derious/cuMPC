#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <algorithm>
// 实现近似计算公式
double approx_exp(double x, int r = 8) {
    if (x > 0) return 0;  // 只处理 x <= 0 的情况
    
    double base = 1.0 + x / (256);
    double result = base;
    
    // 计算2^r次方
    for (int i = 0; i < r; i++) {
        result *= result;
    }
    
    return result;
}

double pade_approximation(double x) {
    if (x > 0) return 0;
    // 使用[2,2] Padé近似
    double num = 1.0 + x/2.0 + x*x/12.0;
    double den = 1.0 - x/2.0 + x*x/12.0;
    return num/den;
}

// 定义查找表结构
struct ExpLookupEntry {
    float x;
    uint8_t q;
};

// 8bit量化的指数函数实现
class QuantizedExp {
private:
    static inline constexpr float kScale = 0.986825f;
    static inline constexpr int kZeroPoint = 0;
    static inline constexpr float kXMin = -8.0f;
    static inline constexpr float kXMax = 5.5f;
    
    // 查找表
    static inline constexpr ExpLookupEntry kLookupTable[] = {
        {-8.00f, 0},
        {-7.10f, 0},
        {-6.19f, 0},
        {-5.29f, 1},
        {-4.38f, 1},
        {-3.48f, 3},
        {-2.57f, 8},
        {-1.67f, 19},
        {-0.76f, 47},
        {0.14f, 116},
        {1.05f, 255},
        {1.95f, 255},
        {2.86f, 255},
        {3.76f, 255},
        {4.67f, 255},
        {5.50f, 255}
    };
    static inline constexpr size_t kTableSize = 16;

public:
    // 添加一个公共方法来获取 scale
    static float get_scale() { return kScale; }
    
    // 使用查找表的量化指数计算
    static uint8_t compute_with_lut(float x) {
        // 输入截断
        x = std::min(std::max(x, kXMin), kXMax);
        
        // 在查找表中找到相邻的两个点
        for (size_t i = 0; i < kTableSize - 1; ++i) {
            if (x >= kLookupTable[i].x && x <= kLookupTable[i + 1].x) {
                // 线性插值
                float x0 = kLookupTable[i].x;
                float x1 = kLookupTable[i + 1].x;
                float q0 = kLookupTable[i].q;
                float q1 = kLookupTable[i + 1].q;
                
                float t = (x - x0) / (x1 - x0);
                return static_cast<uint8_t>(std::round(q0 + t * (q1 - q0)));
            }
        }
        
        // 如果x超出范围，返回边界值
        return (x <= kXMin) ? kLookupTable[0].q : kLookupTable[kTableSize - 1].q;
    }
    
    // 使用直接计算的量化指数
    static uint8_t compute_direct(float x) {
        // 输入截断
        x = std::min(std::max(x, kXMin), kXMax);
        
        // 计算量化值
        float exp_val = std::exp(x);
        int quant_val = std::round(exp_val / kScale + kZeroPoint);
        
        // 确保输出在[0, 255]范围内
        return static_cast<uint8_t>(std::min(std::max(quant_val, 0), 255));
    }
};

int main() {
    // 测试量化指数函数
    std::cout << "\n测试量化指数函数:\n";
    std::cout << "      x      |  量化(LUT)  | 量化(直接)  |    实际值    | 相对误差(%)\n";
    std::cout << "-------------|-------------|-------------|--------------|-------------\n";
    
    for (float x = -8.0f; x <= 5.5f; x += 0.5f) {
        uint8_t quant_lut = QuantizedExp::compute_with_lut(x);
        uint8_t quant_direct = QuantizedExp::compute_direct(x);
        double actual = std::exp(x);
        
        // 计算相对误差 (使用LUT结果)
        double reconstructed = quant_lut * QuantizedExp::get_scale();
        double rel_error = std::abs(reconstructed - actual) / actual * 100;
        
        std::cout << std::setw(11) << x << " | "
                  << std::setw(11) << (int)quant_lut << " | "
                  << std::setw(11) << (int)quant_direct << " | "
                  << std::setw(12) << actual << " | "
                  << std::setw(11) << rel_error << "\n";
    }
    
    return 0;
}