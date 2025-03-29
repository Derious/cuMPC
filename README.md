# High-efficiency Secure Two Party Computation on GPU 

## Specifications

- OS: Linux x64
- Language: C++, CUDA
- Requires: OpenSSL, emp-tools,  emp-ot, Eigen-3.4.0, cuda-12.4

## Overview
This project focuses on the GPU implementation and optimization of non-linear functions in privacy-preserving machine learning (PPML), aiming to prevent semi-honest adversary attacks under the (2+1)-PC setting. The project has implemented GPU-based solutions for distributed point function (DPF ), most significant bit (MSB) extraction function, comparison function, ToPK function, 8-bit privacy LUT function and Matrix Multiplication.
This project includes three main subdirectories:
1. `mpc_cuda`: Implementation of secure two-party computation on GPU.
2. `mpc_keys`: Key definition for FSS and MSB on GPU.
3. `test_cuda`: Test cases for GPU.

## Example Testing

### Prepare
To prepare the environment, into the root dir and do the following: 
```
  $ cd dependencie/emp-tool
  $ mkdir build && cd build
  $ cmake ../
  $ make 
  $ sudo make install
```

```
  $ cd dependencie/emp-ot
  $ mkdir build && cd build
  $ cmake ../
  $ make 
  $ sudo make install
```
```
  $ cd dependencie/eigen-3.4.0
  $ mkdir build && cd build
  $ cmake ../
  $ sudo make install
```

### Compile

To compile and test the msb extraction protocol, into the root dir and do the following: 

```
  $ mkdir build && cd build
  $ cmake ../
  $ make test_test_msb
```
### Running

If you want to test the code in local machine, into the build dir and  type

```
  $../run ./bin/test_test_msb //when nP=2 which means it involves 2 parties
```

If you want to test the code over two machine, into the build dir and type

```
./bin/[binaries] 1 12345  //on one machine and

./bin/[binaries] 2 12345  //on the other.
```

You can modify the IP in the **MPABY_util/util_cmpc_config.h** to communicate with multiple machines

## To do
Add the two-party compution for LLM
