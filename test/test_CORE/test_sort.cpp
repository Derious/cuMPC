#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <chrono>
#include <iomanip>

using namespace std;

// 生成一个随机采样的子集
vector<int> sample(int n, int sample_size) {
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0); // [0, 1, ..., n-1]
    random_device rd;
    mt19937 gen(rd());
    shuffle(indices.begin(), indices.end(), gen);
    return vector<int>(indices.end() - sample_size, indices.end());
}

// 比较函数（模拟比较操作）
bool compare(int a, int b) {
    return a < b; // 比较大小
}

// 对部分数据划分块
vector<vector<int>> partition(const vector<int>& arr, const vector<int>& pivots) {
    vector<vector<int>> blocks(pivots.size() + 1);
    for (int num : arr) {
        bool placed = false;
        for (size_t i = 0; i < pivots.size(); ++i) {
            if (compare(num, pivots[i])) {
                blocks[i].push_back(num);
                placed = true;
                break;
            }
        }
        if (!placed) {
            blocks[pivots.size()].push_back(num);
        }
    }
    return blocks;
}

// AAV86 排序的核心函数
void aav86_sort(vector<int>& arr, int k) {
    int n = arr.size();
    if (k == 1 || n <= 1) {
        sort(arr.begin(), arr.end()); // 基本情况，直接排序
        return;
    }

    // 第一步：计算 p = ceil(n^(1/k))
    int p = ceil(pow(n, 1.0 / k));

    // 第二步：采样 P 集合
    vector<int> sample_indices = sample(n, p - 1);
    vector<int> pivots;
    for (int index : sample_indices) {
        pivots.push_back(arr[index]);
    }
    sort(pivots.begin(), pivots.end()); // 确保采样的锚点已排序

    // 第三步：将数据划分为 p 个块
    vector<int> A, B;
    set<int> pivot_set(sample_indices.begin(), sample_indices.end());
    for (int i = 0; i < n; ++i) {
        if (pivot_set.count(i)) {
            B.push_back(arr[i]); // P 集合的元素
        } else {
            A.push_back(arr[i]); // 剩余元素
        }
    }

    vector<vector<int>> blocks = partition(A, pivots);

    // 第四步：递归排序每个块
    for (vector<int>& block : blocks) {
        aav86_sort(block, k - 1); // 递归调用
    }

    // 第五步：合并所有块和锚点
    arr.clear();
    for (size_t i = 0; i < blocks.size(); ++i) {
        arr.insert(arr.end(), blocks[i].begin(), blocks[i].end());
        if (i < B.size()) {
            arr.push_back(B[i]); // 插入锚点
        }
    }
}

vector<int> generateRandomData(int size, int min_val = 1, int max_val = 0xFFFFFF) {
    vector<int> data(size);
    random_device rd;  // 获取随机数种子
    mt19937 gen(rd()); // 使用Mersenne Twister生成器
    uniform_int_distribution<> dis(min_val, max_val);

    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    return data;
}

// 添加新的函数：AAV86 排序的修改版本 - 只选择最大值
void modified_aav86_max(vector<int>& arr, int k, int* number) {
    int n = arr.size();
    cout << "n = " << n << ", k = " << k << endl;
    
    if (k == 1 || n <= 1) {
        *number += n*(n-1)/2;
        // 基本情况，直接找最大值
        if (n > 1) {
            int max_val = arr[0];
            int max_idx = 0;
            for (int i = 1; i < n; i++) {
                if (arr[i] > max_val) {
                    max_val = arr[i];
                    max_idx = i;
                }
            }
            // 将最大值放到数组末尾
            swap(arr[max_idx], arr[n-1]);
        }
        return;
    }

    // 计算 p = ceil(n^(1/k))
    int p = ceil(pow(n, 1.0 / k));
    // p = 2;
    // int p = 3;
    cout << "p = " << p << endl;
    *number += (n-p-1)*(p-1);
    // 采样 P 集合
    vector<int> sample_indices = sample(n, p - 1);
    vector<int> pivots;
    for (int index : sample_indices) {
        pivots.push_back(arr[index]);
    }
    sort(pivots.begin(), pivots.end());

    // 将数据划分为 A（非采样点）和 B（采样点）
    vector<int> A;
    set<int> pivot_set(sample_indices.begin(), sample_indices.end());
    for (int i = 0; i < n; ++i) {
        if (!pivot_set.count(i)) {
            A.push_back(arr[i]);
        }
    }

    // 只对 A 进行划分，找出大于最大枢轴值的块
    vector<int> larger_block;
    for (int num : A) {
        if (pivots.empty() || num > pivots.back()) {
            larger_block.push_back(num);
        }
    }

    // 只对大于枢轴的块进行递归
    if (!larger_block.empty()) {
        modified_aav86_max(larger_block, k - 1, number);
        
        // 将找到的最大值放回原数组末尾
        arr[n-1] = larger_block.back();
    } else if (!pivots.empty()) {
        // 如果没有更大的元素，则最大枢轴值就是最大值
        arr[n-1] = pivots.back();
    }
}

// 快速选择算法实现的TopK选择
vector<int> quick_select_topk(vector<int>& arr, int K, int* number) {
    if (arr.empty() || K <= 0) return vector<int>();
    printf("arr size: %lu, K: %d\n", arr.size(), K);
    // 选择最后一个元素作为pivot
    int pivot = arr.back();
    vector<int> S_L, S_R;
    S_R.push_back(pivot);  // pivot直接放入右侧集合

    // 将其他元素分到左右两个集合
    for (size_t i = 0; i < arr.size() - 1; i++) {
        (*number)++;  // 记录比较次数
        if (arr[i] >= pivot) {
            S_R.push_back(arr[i]);
        } else {
            S_L.push_back(arr[i]);
        }
    }

    int K_prime = S_R.size();

    // 根据K'和K的关系决定下一步操作
    if (K_prime == K) {
        return S_R;
    } else if (K_prime > K) {
        return quick_select_topk(S_R, K, number);
    } else {  // K_prime < K
        vector<int> remaining = quick_select_topk(S_L, K - K_prime, number);
        remaining.insert(remaining.end(), S_R.begin(), S_R.end());
        return remaining;
    }
}

int main() {
    // vector<int> data = {5, 2, 9, 1, 5, 6, 10, 3, 8, 7};
    vector<int> data = generateRandomData(10720);
    vector<int> data2 = data;

    int n = data.size();
    int k = 6; // 设置迭代次数
     
    //Shuffle the values of data
    random_shuffle(data.begin(), data.end());

    int number = 0;
    auto start = chrono::high_resolution_clock::now();
    modified_aav86_max(data, k, &number);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for finding max: " << duration.count() << " milliseconds" << endl;
    cout << "max value: " << data.back() << endl;
    cout << "compare number: " << number << endl;

    start = chrono::high_resolution_clock::now();
    auto max_value = std::max_element(data.begin(), data.end());
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for finding max: " << duration.count() << " milliseconds" << endl;
    cout << "Maximum value: " << *max_value << endl;

    // 测试快速选择的TopK实现
    // vector<int> data_quickselect = generateRandomData(30720);
    int topk = 1;  // 要找前100个最大值
    number = 0;
    
    start = chrono::high_resolution_clock::now();
    vector<int> result = quick_select_topk(data2, topk, &number);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "\nQuick Select Top " << topk << " values:" << endl;
    for (int i = 0; i < min(topk, (int)result.size()); ++i) {
        cout << result[i] << " ";
    }
    cout << "\nTime taken for finding top " << topk << ": " 
         << duration.count() << " milliseconds" << endl;
    cout << "Compare number: " << number << endl;

    return 0;
}

// ... existing code ...

// int main() {
//     const int SIZE = 10;  // 使用更大的数据集以更好地比较性能
//     const int K = 3;  // 设置迭代次数

//     // 测试原始 AAV86 排序
//     vector<int> data1 = generateRandomData(SIZE);
//     cout << "Testing original AAV86 sort with " << SIZE << " elements:" << endl;
//     cout << "First 10 elements before sorting: ";
//     for (int i = 0; i < 10; i++) {
//         cout << data1[i] << " ";
//     }
//     cout << "..." << endl;

//     // 计时开始
//     auto start1 = chrono::high_resolution_clock::now();
    
//     aav86_sort(data1, K);
    
//     // 计时结束
//     auto end1 = chrono::high_resolution_clock::now();
//     auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);

//     cout << "First 10 elements after sorting: ";
//     for (int i = 0; i < 10; i++) {
//         cout << data1[i] << " ";
//     }
//     cout << "..." << endl;
//     cout << "Time taken for full sort: " << duration1.count() << " milliseconds" << endl;
//     cout << endl;

//     // 测试修改后的最大值查找
//     vector<int> data2 = generateRandomData(SIZE);
//     cout << "Testing modified AAV86 max with " << SIZE << " elements:" << endl;
//     cout << "First 10 elements before finding max: ";
//     for (int i = 0; i < 10; i++) {
//         cout << data2[i] << " ";
//     }
//     cout << "..." << endl;

//     // 计时开始
//     auto start2 = chrono::high_resolution_clock::now();
    
//     modified_aav86_max(data2, K);
    
//     // 计时结束
//     auto end2 = chrono::high_resolution_clock::now();
//     auto duration2 = chrono::duration_cast<chrono::milliseconds>(end2 - start2);

//     cout << "Maximum value: " << data2.back() << endl;
//     cout << "Time taken for finding max: " << duration2.count() << " milliseconds" << endl;
//     cout << endl;

//     // 比较性能
//     cout << "Performance Comparison:" << endl;
//     cout << "Full sort time: " << duration1.count() << " ms" << endl;
//     cout << "Find max time: " << duration2.count() << " ms" << endl;
//     cout << "Speed improvement: " << fixed << setprecision(2) 
//          << (float)duration1.count() / duration2.count() << "x" << endl;

//     return 0;
// }