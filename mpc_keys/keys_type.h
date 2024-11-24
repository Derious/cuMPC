#ifndef KEYS_TYPE_H
#define KEYS_TYPE_H

#include <Eigen/Dense>
using namespace Eigen;


// 使用 RowMajor 存储格式
using MatrixRowMajor = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct MM_Keys {
    MatrixRowMajor R_A;
    MatrixRowMajor R_B;
    MatrixRowMajor R_AB;

    MM_Keys(int rowsA, int colsA, int colsB) : R_A(rowsA, colsA), R_B(colsA, colsB), R_AB(rowsA, colsB) {
        R_A.setZero();
        R_B.setZero();
        R_AB.setZero(); 
    }  
};

typedef unsigned char *DCF_Keys;

#endif // KEYS_TYPE_H