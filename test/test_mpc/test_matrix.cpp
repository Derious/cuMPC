#include <Eigen/Dense>
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "mpc_cuda/mpc_core.h"
#include "../../MPABY_GMW/GMW_protocol.h"
using namespace emp;
using namespace Eigen;
using namespace std;


#define BENCH 20
const static int nP = 2;
int party, port;
const static int length = 256*768;
const static int64_t rowsA = 1, colsA = 65536, rowsB = 65536, colsB = 1;

int main(int argc, char** argv) {
 
	parse_party_and_port(argv, &party, &port);
    if(party > nP)return 0;
	printf("party:%d	port:%d\n",party,port);
	

	NetIOMP<nP> io(party, port);
	ThreadPool pool(4);	
    PRG prg;
	GMWprotocolA<nP>* gmw = new GMWprotocolA<nP>(&io,&pool,party);
    cuda_mpc_core<nP> *cuda_mpc = new cuda_mpc_core<nP>(&io, &pool, party);
    cudaWarmup(1024 * 1024, party);

    // MatrixRowMajor matA(rowsA, colsA);
    // matA.setOnes();

	// auto start = clock_start();
    // gmw->open_vec(matA.data(),matA.data(),matA.size());
	// double timeused = time_from(start);
	// cout << "ReLU time used: " << timeused / 1000 << " ms" << endl;
    // if (party == 1)
    // {
    //     uint64_t band2 = io.count();
	//     cout <<"EMPDM_ReLU bandwidth\t"<<party<<"\t"<<band2<<endl;
    //     cout << "matA: " << matA << endl;
    // }


    // 矩阵尺寸
    
    // 定义矩阵 (RowMajor 格式)
    MatrixRowMajor matA(rowsA, colsA);
    MatrixRowMajor matB(rowsB, colsB);

    //random init and set random seed
    srand(time(NULL));
    matA.setRandom();
    matB.setRandom();

    MM_Keys keys(rowsA, colsA, colsB);

    if(party == 1) {
        keys.R_A.setRandom();
        keys.R_B.setRandom();
        keys.R_AB = keys.R_A * keys.R_B;
    }

    MatrixRowMajor Public_A(rowsA, colsA);
    MatrixRowMajor Public_B(rowsB, colsB);
    Public_A.setZero();
    Public_B.setZero();

    Public_A = matA + keys.R_A;
    Public_B = matB + keys.R_B;
    gmw->open_vec_2PC(Public_B.data(), Public_B.data(), Public_B.size());
    gmw->open_vec_2PC(Public_A.data(), Public_A.data(), Public_A.size());
    

    gmw->open_vec_2PC(matA.data(), matA.data(), matA.size());
    gmw->open_vec_2PC(matB.data(), matB.data(), matB.size());


    // // 打印输入矩阵
    // std::cout << "Matrix A:\n" << matA << "\n\n";
    // std::cout << "Matrix B:\n" << matB << "\n\n";

    // 结果矩阵
    MatrixRowMajor matC(rowsA, colsB);
    matC.setZero();

    MatrixRowMajor matC2(rowsA, colsB);
    matC2.setZero();
    // // 调用 CUDA 矩阵乘法
    // auto start = clock_start();
    // cudaMatrixMultiply(matA.data(), matB.data(), matC.data(), rowsA, colsA, colsB);
    // double timeused  = time_from(start);

    // cuda_MM(matC, matA, matB);
    auto start = clock_start();
    for(int i = 0; i < BENCH; i++)
        cuda_mpc->CUDA_MPC_MM(matC, Public_A, Public_B, keys, party);
    double timeused  = time_from(start);
    std::cout << "CUDA Time taken: " << timeused / (1000*BENCH) << " ms" << std::endl;
    // cuda_Linear(matC2, 5, matA, 1, matA, 10);
    auto start2 = clock_start();
    matC2 = matA*matB;
    double timeused2  = time_from(start2);
    std::cout << "CPU Time taken: " << timeused2 / (1000) << " ms" << std::endl;
    // cout << "matC: " << matC << endl;
    // cout << "true result: " << matC2 << endl;
    // cout << "R_A: " << Public_A << endl;
    // cout << "R_B: " << Public_B << endl;

    return 0;
}
