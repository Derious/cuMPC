#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
using namespace Eigen;
#define SIZE atoi(argv[1])
using MatrixRowMajor = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// 随机数生成器
std::mt19937 rng(std::random_device{}());

// 随机选择 pivots
std::vector<int> selectPivots(const std::vector<int>& arr, int numPivots) {
    std::vector<int> pivots;
    // std::sample(arr.begin(), arr.end(), std::back_inserter(pivots), numPivots, rng);
    pivots.insert(pivots.end(), arr.end()-numPivots, arr.end());
    std::sort(pivots.begin(), pivots.end()); // 按顺序排列 pivots
    return pivots;
}

// 分区函数：根据 pivots 将数组分成多个区块
std::vector<std::vector<int>> partition(const std::vector<int>& arr, const std::vector<int>& pivots) {

    // Map<MatrixRowMajor> value_map((int*)arr.data(), 1, arr.size());
    // MatrixRowMajor value_mat = value_map.replicate(pivots.size(), 1);
    // std::cout << "value_mat: \n" << value_mat << std::endl;
    // Map<Vector<int,Dynamic>> pivots_map((int*)pivots.data(), pivots.size());
    // value_mat = value_mat.colwise() - pivots_map;
    // std::cout << "pivots_map: \n" << pivots_map << std::endl;
    // std::cout << "value_mat: \n" << value_mat << std::endl;
    // MatrixRowMajor result = value_mat.colwise().sum();
    // std::cout << "result: \n" << result << std::endl;

    std::vector<std::vector<int>> blocks(pivots.size() + 1); // pivots.size() + 1 个区块
    for (int val : arr) {
        int i = 0;
        while (i < pivots.size() && val > pivots[i]) {
            i++;
        }
        blocks[i].push_back(val);
    }
    return blocks;
}

// TopK 主函数
void topKSelection(std::vector<int>& result, std::vector<int>& arr, int K, int maxIterations) {
    int n = arr.size();
    int k = maxIterations;
    result.clear();



    while (k > 0) {
        printf("K: %d\n", K);
        printf("input size: %d\n", arr.size());
        // for(int i = 0; i < arr.size(); i++){
        //     printf("%d ", arr[i]);
        // }
        // printf("\n");
        // 确定 pivots 数量
        int p = std::ceil(std::pow(n, 1.0 / k));
        p = std::min(p, n); // 防止 pivots 数量超过数组大小
        printf("p: %d\n", p);

        // 选择 pivots
        std::vector<int> pivots = selectPivots(arr, p - 1);
        // printf("pivots: ");
        // for(int i = 0; i < pivots.size(); i++){
        //     printf("%d ", pivots[i]);
        // }
        // printf("\n");

        // 根据 pivots 对数组进行分区
        std::vector<std::vector<int>> blocks = partition(arr, pivots);
        printf("blocks: ");
        for(int i = 0; i < blocks.size(); i++){
            printf("[");
            printf("%d", blocks[i].size());
            printf("] ");
        }
        printf("\n");

        // 统计区块大小，找到包含 TopK 的区块
        int total = 0;
        int targetBlock = -1;
        arr.clear();
        for (int i = blocks.size() - 1; i >= 0; i--) {
            if (total + blocks[i].size() >= K) {
                arr.insert(arr.end(), blocks[i].begin(), blocks[i].end());
                break;
            }
            result.insert(result.end(), blocks[i].begin(), blocks[i].end());
            total += blocks[i].size();
        }
        // printf("result: ");
        // for(int i = 0; i < result.size(); i++){
        //     printf("%d ", result[i]);
        // }
        // printf("\n"); 
        // 更新 K 和 arr
        K -= total;
        n = arr.size();

        // 如果当前区块大小小于等于 K，直接返回
        if (n <= K) {
            std::sort(arr.begin(), arr.end()); // 对最后的区块排序
            arr.resize(K); // 保留前 K 个元素
            result.insert(result.end(), arr.begin(), arr.end());
            return;
        }

        // 迭代次数减少
        k--;
    }

    // 最后一步：排序并返回前 K 个元素
    printf("result size: %ld\n", arr.size());
    std::sort(arr.begin(), arr.end());
    result.insert(result.end(), arr.end()-K, arr.end());
}

void RankSort(std::vector<int>& result, std::vector<int>& arr){
    std::vector<int> rank(arr.size());
    for(size_t i = 0; i < arr.size(); i++){
        for(size_t j = 0; j < arr.size(); j++){
            if(arr[i] > arr[j]){
                rank[i]++;
            }
        }
    }

    for(size_t i = 0; i < arr.size(); i++){
        // printf("rank[%d]: %d\n", i, rank[i]);
        result[rank[i]] = arr[i];
    }
}


// 构造比较矩阵
void MatrixRank(std::vector<int>& result,const std::vector<int>& arr) {
    size_t n = arr.size();
    std::vector<int> rank(arr.size());
    Eigen::MatrixXi matrix = Eigen::MatrixXi::Zero(n, n);

    // 仅计算上三角矩阵部分
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            matrix(i, j) = arr[i] > arr[j];
            matrix(j, i) = 1 - matrix(i, j);
        }
    }
    // 每行的和即为排名值
    for (int i = 0; i < n; ++i) {
        rank[i] = matrix.row(i).sum();
        result[rank[i]] = arr[i];
    }
}


void TopKselection(std::vector<int>& result, std::vector<int>& arr, int K, int &iter){
    int n = arr.size();
    result.clear();
    while (K > 0) {
        printf("K: %d\n", K);
        printf("input size: %d\n", arr.size());
        // for(int i = 0; i < arr.size(); i++){
        //     printf("%d ", arr[i]);
        // }
        // printf("\n");
        // 确定 pivots 数量
        int p = 2;

        // 选择 pivots
        std::vector<int> pivots = selectPivots(arr, p - 1);
        arr.pop_back();
        // printf("pivots: ");
        // for(int i = 0; i < pivots.size(); i++){
        //     printf("%d ", pivots[i]);
        // }
        // printf("\n");

        // 根据 pivots 对数组进行分区
        std::vector<std::vector<int>> blocks = partition(arr, pivots);
        printf("blocks: ");
        for(int i = 0; i < blocks.size(); i++){
            printf("[");
            printf("%d", blocks[i].size());
            printf("] ");
        }
        printf("\n");

        // 统计区块大小，找到包含 TopK 的区块
        int total = 0;
        arr.clear();
        for (int i = blocks.size() - 1; i >= 0; i--) {
            if (total + blocks[i].size() >= K) {
                arr.insert(arr.end(), blocks[i].begin(), blocks[i].end());
                break;
            }
            result.insert(result.end(), blocks[i].begin(), blocks[i].end());
            result.insert(result.end(), pivots.begin(), pivots.end());
            total += blocks[i].size() + pivots.size();
        }
        // printf("result: ");
        // for(int i = 0; i < result.size(); i++){
        //     printf("%d ", result[i]);
        // }
        // printf("\n"); 
        // 更新 K 和 arr
        K -= total;
        n = arr.size();

        //remove pivots from arr
        

        // 如果当前区块大小小于等于 K，直接返回
        if (n <= K) {
            std::sort(arr.begin(), arr.end()); // 对最后的区块排序
            arr.resize(K); // 保留前 K 个元素
            result.insert(result.end(), arr.begin(), arr.end());
            return;
        }

        // 迭代次数减少
        // k--;
        iter++;
    }

    // 最后一步：排序并返回前 K 个元素
    printf("result size: %ld\n", arr.size());
    std::sort(arr.begin(), arr.end());
    result.insert(result.end(), arr.end()-K, arr.end());
}

int main(int argc, char* argv[]) {
    // 测试数组
    //sample 1000 random numbers
    srand(time(NULL));
    std::vector<int> input(SIZE);
    std::vector<int> arr;
    std::vector<int> arr2;
    std::generate(input.begin(), input.end(), std::rand);
    arr.insert(arr.end(), input.begin(), input.end());
    arr2.insert(arr2.end(), input.begin(), input.end());
    std::shuffle(arr.begin(), arr.end(), rng);
    int K = atoi(argv[2]); // 找出前 5 个元素
    int maxIterations = atoi(argv[3]); // 设置最大迭代次数
    std::vector<int> result;
    std::cout << "TopKselection" << std::endl;
    topKSelection(result, arr, K, maxIterations);
    std::cout << "=============================================" << std::endl;
    std::cout << "TopKselection" << std::endl;
    int iter = 0;
    TopKselection(result, arr2, K, iter);
    std::cout << "iter: " << iter << std::endl;
    std::cout << "=============================================" << std::endl;
    // 输出结果
    std::sort(result.begin(), result.end());
    // std::cout << "Top " << K << " elements: ";
    // for (int val : result) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    //验证结果
    std::sort(input.begin(), input.end());
    std::cout << "Top " << K << " elements: ";
    for (int i = SIZE-K; i <SIZE; i++) {
        if(result[i-SIZE+K] != input[i]){
            printf("error: %d %d\n", result[i-SIZE+K], input[i]);
        }
        // std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    // std::vector<int> result2(SIZE);
    // MatrixRank(result2, arr2);
    // std::cout << "Top " << K << " elements: ";
    // for (int i = SIZE-K; i <SIZE; i++) {
    //     std::cout << result2[i] << " ";
    // }
    // std::cout << std::endl;
    
    // std::shuffle(input.begin(), input.end(), rng);
    // int64_t* value = new int64_t[SIZE];
    // for(int i = 0; i < SIZE; i++){
    //     value[i] = input[i];
    //     printf("%d ", value[i]);
    // }
    // printf("\n");
    // Map<MatrixRowMajor> value_map(value, 1, SIZE);

    // // std::cout << value_map << std::endl;
    // MatrixRowMajor value_mat = value_map.replicate(SIZE, 1);
    // std::cout << "value_mat: " << std::endl;
    // std::cout << value_mat << std::endl;
    // std::cout << "value_mat.transpose(): " << std::endl;
    // MatrixRowMajor value_mat_transpose = value_mat.transpose();
    // std::cout << value_mat_transpose << std::endl;
    // value_mat = (value_mat_transpose - value_mat);
    // std::cout << "value_mat - value_mat.transpose(): " << std::endl;
    // value_mat = value_mat.triangularView<StrictlyUpper>();
    // std::cout << value_mat << std::endl;

    // for(int i = 0; i < SIZE*SIZE; i++){
    //     printf("%d ", value_mat.array()(i));
    // }
    // printf("\n");

    return 0;
}
