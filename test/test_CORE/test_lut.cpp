#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>

// 量化参数（由Python计算得出）
const float SCALE = 34.011650f;  // 替换为实际的scale值
const float ZERO_POINT = 0.0f;  // 替换为实际的zero_point值

const float GeLU_SCALE =  1/0.020573f;
const float GeLU_ZERO_POINT = 214.0f;


const int64_t FIXED_POINT_SCALE = (1<<20);
const int64_t FIXED_BITS = 20;

const int64_t GeLU_SCALE_FIXED = 1/0.020573f * FIXED_POINT_SCALE;
const int64_t GeLU_ZERO_POINT_FIXED = 214.0f * FIXED_POINT_SCALE;



inline float fixed_to_float(int64_t fixed) {
    return fixed / (float)FIXED_POINT_SCALE;
}

inline int64_t float_to_fixed(float f) {
    return (f * FIXED_POINT_SCALE);
}

float GeLU1(float x) {
    return 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
}

//def gelu(x):return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x, 3))))
float GeLU(float x) {
    return 0.5 * x * (1 + std::tanh(std::sqrt(2/M_PI)*(x+0.044715*std::pow(x, 3))));
}

const float rsqrt_lut[256] = {
    0.496319f, 0.162070f, 0.117783f, 0.097085f, 0.084483f, 0.075784f, 0.069316f, 0.064264f,
    0.060176f, 0.056781f, 0.053903f, 0.051422f, 0.049255f, 0.047340f, 0.045633f, 0.044098f,
    0.042708f, 0.041442f, 0.040282f, 0.039215f, 0.038228f, 0.037312f, 0.036459f, 0.035661f,
    0.034914f, 0.034212f, 0.033551f, 0.032927f, 0.032336f, 0.031776f, 0.031244f, 0.030738f,
    0.030255f, 0.029795f, 0.029355f, 0.028934f, 0.028531f, 0.028144f, 0.027772f, 0.027415f,
    0.027071f, 0.026740f, 0.026421f, 0.026113f, 0.025815f, 0.025527f, 0.025249f, 0.024980f,
    0.024719f, 0.024466f, 0.024221f, 0.023982f, 0.023751f, 0.023527f, 0.023308f, 0.023096f,
    0.022889f, 0.022688f, 0.022492f, 0.022301f, 0.022115f, 0.021933f, 0.021756f, 0.021583f,
    0.021414f, 0.021249f, 0.021087f, 0.020930f, 0.020775f, 0.020625f, 0.020477f, 0.020333f,
    0.020191f, 0.020053f, 0.019917f, 0.019784f, 0.019653f, 0.019526f, 0.019400f, 0.019277f,
    0.019157f, 0.019038f, 0.018922f, 0.018808f, 0.018696f, 0.018585f, 0.018477f, 0.018371f,
    0.018266f, 0.018164f, 0.018062f, 0.017963f, 0.017865f, 0.017769f, 0.017674f, 0.017581f,
    0.017490f, 0.017399f, 0.017310f, 0.017223f, 0.017137f, 0.017052f, 0.016968f, 0.016886f,
    0.016804f, 0.016724f, 0.016645f, 0.016567f, 0.016491f, 0.016415f, 0.016340f, 0.016266f,
    0.016194f, 0.016122f, 0.016051f, 0.015981f, 0.015912f, 0.015844f, 0.015777f, 0.015711f,
    0.015645f, 0.015580f, 0.015517f, 0.015453f, 0.015391f, 0.015329f, 0.015268f, 0.015208f,
    0.015149f, 0.015090f, 0.015032f, 0.014975f, 0.014918f, 0.014862f, 0.014806f, 0.014751f,
    0.014697f, 0.014643f, 0.014590f, 0.014538f, 0.014486f, 0.014434f, 0.014383f, 0.014333f,
    0.014283f, 0.014234f, 0.014185f, 0.014137f, 0.014089f, 0.014042f, 0.013995f, 0.013948f,
    0.013903f, 0.013857f, 0.013812f, 0.013767f, 0.013723f, 0.013680f, 0.013636f, 0.013593f,
    0.013551f, 0.013509f, 0.013467f, 0.013426f, 0.013385f, 0.013344f, 0.013304f, 0.013264f,
    0.013224f, 0.013185f, 0.013146f, 0.013108f, 0.013070f, 0.013032f, 0.012995f, 0.012957f,
    0.012921f, 0.012884f, 0.012848f, 0.012812f, 0.012776f, 0.012741f, 0.012706f, 0.012671f,
    0.012637f, 0.012603f, 0.012569f, 0.012535f, 0.012502f, 0.012469f, 0.012436f, 0.012403f,
    0.012371f, 0.012339f, 0.012307f, 0.012275f, 0.012244f, 0.012213f, 0.012182f, 0.012151f,
    0.012121f, 0.012091f, 0.012061f, 0.012031f, 0.012002f, 0.011972f, 0.011943f, 0.011914f,
    0.011886f, 0.011857f, 0.011829f, 0.011801f, 0.011773f, 0.011746f, 0.011718f, 0.011691f,
    0.011664f, 0.011637f, 0.011610f, 0.011584f, 0.011557f, 0.011531f, 0.011505f, 0.011479f,
    0.011454f, 0.011428f, 0.011403f, 0.011378f, 0.011353f, 0.011328f, 0.011303f, 0.011279f,
    0.011255f, 0.011230f, 0.011206f, 0.011183f, 0.011159f, 0.011135f, 0.011112f, 0.011089f,
    0.011066f, 0.011043f, 0.011020f, 0.010997f, 0.010975f, 0.010952f, 0.010930f, 0.010908f,
    0.010886f, 0.010864f, 0.010842f, 0.010820f, 0.010799f, 0.010778f, 0.010756f, 0.010735f,
};


const float gelu_lut[256] = {
    -0.000009f, -0.000010f, -0.000011f, -0.000013f, -0.000014f, -0.000016f, -0.000017f, -0.000019f,
    -0.000022f, -0.000024f, -0.000027f, -0.000030f, -0.000033f, -0.000037f, -0.000041f, -0.000045f,
    -0.000050f, -0.000055f, -0.000061f, -0.000067f, -0.000074f, -0.000081f, -0.000090f, -0.000099f,
    -0.000109f, -0.000119f, -0.000131f, -0.000144f, -0.000158f, -0.000173f, -0.000189f, -0.000207f,
    -0.000227f, -0.000248f, -0.000270f, -0.000295f, -0.000322f, -0.000350f, -0.000382f, -0.000415f,
    -0.000452f, -0.000491f, -0.000533f, -0.000578f, -0.000627f, -0.000679f, -0.000736f, -0.000796f,
    -0.000861f, -0.000931f, -0.001006f, -0.001086f, -0.001171f, -0.001263f, -0.001360f, -0.001465f,
    -0.001576f, -0.001695f, -0.001821f, -0.001956f, -0.002099f, -0.002252f, -0.002414f, -0.002586f,
    -0.002769f, -0.002962f, -0.003168f, -0.003386f, -0.003616f, -0.003860f, -0.004118f, -0.004391f,
    -0.004678f, -0.004982f, -0.005303f, -0.005641f, -0.005996f, -0.006371f, -0.006765f, -0.007179f,
    -0.007615f, -0.008072f, -0.008552f, -0.009056f, -0.009583f, -0.010136f, -0.010715f, -0.011320f,
    -0.011954f, -0.012615f, -0.013306f, -0.014028f, -0.014780f, -0.015564f, -0.016381f, -0.017232f,
    -0.018117f, -0.019038f, -0.019994f, -0.020988f, -0.022019f, -0.023089f, -0.024198f, -0.025346f,
    -0.026536f, -0.027767f, -0.029039f, -0.030354f, -0.031713f, -0.033115f, -0.034560f, -0.036051f,
    -0.037586f, -0.039166f, -0.040791f, -0.042462f, -0.044179f, -0.045941f, -0.047749f, -0.049603f,
    -0.051502f, -0.053445f, -0.055434f, -0.057467f, -0.059543f, -0.061662f, -0.063824f, -0.066027f,
    -0.068270f, -0.070552f, -0.072873f, -0.075230f, -0.077622f, -0.080048f, -0.082505f, -0.084993f,
    -0.087509f, -0.090050f, -0.092615f, -0.095201f, -0.097806f, -0.100426f, -0.103060f, -0.105703f,
    -0.108353f, -0.111006f, -0.113659f, -0.116308f, -0.118950f, -0.121580f, -0.124195f, -0.126789f,
    -0.129359f, -0.131901f, -0.134408f, -0.136877f, -0.139303f, -0.141680f, -0.144003f, -0.146266f,
    -0.148465f, -0.150593f, -0.152645f, -0.154615f, -0.156496f, -0.158283f, -0.159970f, -0.161549f,
    -0.163015f, -0.164362f, -0.165582f, -0.166670f, -0.167618f, -0.168419f, -0.169068f, -0.169558f,
    -0.169881f, -0.170030f, -0.170000f, -0.169784f, -0.169374f, -0.168764f, -0.167947f, -0.166917f,
    -0.165667f, -0.164191f, -0.162482f, -0.160535f, -0.158342f, -0.155899f, -0.153198f, -0.150235f,
    -0.147003f, -0.143498f, -0.139713f, -0.135644f, -0.131286f, -0.126634f, -0.121684f, -0.116431f,
    -0.110871f, -0.105000f, -0.098815f, -0.092312f, -0.085488f, -0.078341f, -0.070867f, -0.063064f,
    -0.054930f, -0.046464f, -0.037664f, -0.028528f, -0.019055f, -0.009246f, 0.000901f, 0.011386f,
    0.022209f, 0.033368f, 0.044864f, 0.056694f, 0.068859f, 0.081357f, 0.094184f, 0.107341f,
    0.120823f, 0.134628f, 0.148754f, 0.163197f, 0.177954f, 0.193021f, 0.208394f, 0.224069f,
    0.240041f, 0.256307f, 0.272861f, 0.289698f, 0.306814f, 0.324203f, 0.341859f, 0.359776f,
    0.377950f, 0.396373f, 0.415041f, 0.433946f, 0.453083f, 0.472445f, 0.492026f, 0.511818f,
    0.531817f, 0.552014f, 0.572403f, 0.592978f, 0.613731f, 0.634656f, 0.655747f, 0.676996f,
};

int64_t gelu_lut_fixed[256] = {0};
// LUT查找函数
inline float rsqrt_lookup(float x) {
    // 量化
    int index = std::round(x/SCALE + ZERO_POINT);
    if (index < 0) index = 0;
    if (index > 255) index = 255;
    // 查表
    return rsqrt_lut[index];
}

inline void gelu_lookup_test(int64_t x, float y) {
    // 量化
    int index_fixed = ((x * GeLU_SCALE_FIXED >> FIXED_BITS) + GeLU_ZERO_POINT_FIXED) >> FIXED_BITS;
    printf("index_fixed: %f\n", fixed_to_float(gelu_lut_fixed[index_fixed]));
    int index = std::round(y*GeLU_SCALE + GeLU_ZERO_POINT);
    printf("index: %f\n", gelu_lut[index]);
}

inline int64_t gelu_lookup(int64_t x) {
    // 量化
    int index = ((x * GeLU_SCALE_FIXED >> FIXED_BITS) + GeLU_ZERO_POINT_FIXED) >> FIXED_BITS;
    if (index < 0) index = 0;
    if (index > 255) index = 255;
    // 查表
    return gelu_lut_fixed[index];
}



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


void verify_gelu_lut() {
    std::cout << "Verifying GeLU LUT conversion:\n";
    for (int i = 0; i < 256; i ++) {  // 每32个值检查一个
        int64_t fixed_val = gelu_lut[i] * FIXED_POINT_SCALE;
        gelu_lut_fixed[i] = fixed_val;
        printf("index: %d, val: %f, fixed_val_int: %ld\n", i, gelu_lut[i], gelu_lut_fixed[i]);
    }
}

int main() {
    // verify_gelu_lut();
    int64_t size = (int64_t)0xFFFFFFFF;
    int64_t res = 0;
    auto start = std::chrono::high_resolution_clock::now();
    int64_t i = 0;
    // #pragma omp parallel for reduction(+:res)
    // for(int64_t j = 0; j < 0xF; j++) {
    for (i = 0; i < size; i++)
        {
            res += size;
            /* code */
        }
    // }
    printf("res: %lx\n", res);
    printf("i: %lx\n", i);
    // std::cout << res << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    
    return 0;
}

// int main() {
//     verify_gelu_lut();
//     // 批量测试
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     float min_value = 4.05955f;
//     float max_value = 8677.030176f;
    
//     float log_min = std::log(min_value);
//     float log_max = std::log(max_value);
//     std::uniform_real_distribution<float> log_dist(log_min, log_max);
    
//     // 生成100个测试值
//     std::vector<float> test_values;
//     test_values.reserve(100);

//     std::vector<int64_t> test_values_fixed;
//     test_values_fixed.reserve(100);
    
//     for (int i = 0; i < 100; ++i) {
//         float value = std::exp(log_dist(gen));
//         test_values.push_back(value);
//         test_values_fixed.push_back(float_to_fixed(value));
//     }

//     // for(int i = 0; i < 10; i++) {
//     //     gelu_lookup_test(test_values_fixed[i], test_values[i]);
//     // }

    

//     // 计算误差统计
//     float total_abs_error = 0.0f;
//     float total_rel_error = 0.0f;
//     float max_rel_error = 0.0f;
//     float min_rel_error = std::numeric_limits<float>::max();
    
//     std::cout << "\nBatch test:\n";
//     std::cout << std::setw(15) << "Input" 
//               << std::setw(15) << "LUT Result" 
//               << std::setw(15) << "True Result" 
//               << std::setw(15) << "Rel Error(%)" << std::endl;
//     std::cout << std::string(60, '-') << std::endl;

//     for(float x : test_values) {
//         // std::cout << "Input: " << x 
//                 //   << ", rsqrt: " << rsqrt_lookup(x) << ", true_result: " << 1.0f / std::sqrt(x) << std::endl;
//         float lut_result = rsqrt_lookup(x);
//         float true_val = 1.0f / (std::sqrt(x) + 1e-6);
        
//         float abs_error = std::abs(lut_result - true_val);
//         float rel_error = (abs_error / true_val) * 100.0f;  // 相对误差百分比
        
//         total_abs_error += abs_error;
//         total_rel_error += rel_error;
//         max_rel_error = std::max(max_rel_error, rel_error);
//         min_rel_error = std::min(min_rel_error, rel_error);

//         std::cout << std::fixed << std::setprecision(6)
//                   << std::setw(15) << x 
//                   << std::setw(15) << lut_result
//                   << std::setw(15) << true_val
//                   << std::setw(15) << rel_error << std::endl;
//     }

//     // 输出统计结果
//     std::cout << "\nError Statistics:\n";
//     std::cout << "Average Absolute Error: " << total_abs_error / 100.0f << std::endl;
//     std::cout << "Average Relative Error: " << total_rel_error / 100.0f << "%" << std::endl;
//     std::cout << "Maximum Relative Error: " << max_rel_error << "%" << std::endl;
//     std::cout << "Minimum Relative Error: " << min_rel_error << "%" << std::endl;


//     // // 计算误差统计
//     // total_abs_error = 0.0f;
//     // total_rel_error = 0.0f;
//     // max_rel_error = 0.0f;
//     // min_rel_error = std::numeric_limits<float>::max();
    
//     // std::cout << "\nBatch test:\n";
//     // std::cout << std::setw(15) << "Input" 
//     //           << std::setw(15) << "LUT Result" 
//     //           << std::setw(15) << "True Result" 
//     //           << std::setw(15) << "Rel Error(%)" << std::endl;
//     // std::cout << std::string(60, '-') << std::endl;

//     // for(float x : test_values) {
//     //     // std::cout << "Input: " << x 
//     //             //   << ", rsqrt: " << rsqrt_lookup(x) << ", true_result: " << 1.0f / std::sqrt(x) << std::endl;
//     //     // float gelu_result = gelu_lookup(x);
//     //     float gelu_result = fixed_to_float(gelu_lookup(float_to_fixed(x)));
//     //     float true_val = GeLU(x);
        
//     //     float abs_error = std::abs(gelu_result - true_val);
//     //     float rel_error = (abs_error / true_val) * 100.0f;  // 相对误差百分比
        
//     //     total_abs_error += abs_error;
//     //     total_rel_error += rel_error;
//     //     max_rel_error = std::max(max_rel_error, rel_error);
//     //     min_rel_error = std::min(min_rel_error, rel_error);

//     //     std::cout << std::fixed << std::setprecision(6)
//     //               << std::setw(15) << x 
//     //               << std::setw(15) << gelu_result
//     //               << std::setw(15) << true_val
//     //               << std::setw(15) << rel_error << std::endl;
//     // }

//     // // 输出统计结果
//     // std::cout << "\nError Statistics:\n";
//     // std::cout << "Average Absolute Error: " << total_abs_error / 100.0f << std::endl;
//     // std::cout << "Average Relative Error: " << total_rel_error / 100.0f << "%" << std::endl;
//     // std::cout << "Maximum Relative Error: " << max_rel_error << "%" << std::endl;
//     // std::cout << "Minimum Relative Error: " << min_rel_error << "%" << std::endl;
//     return 0;
// }