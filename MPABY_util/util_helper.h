#ifndef __HELPER
#define __HELPER
#include <emp-tool/emp-tool.h>
#include "util_cmpc_config.h"
#include "util_netmp.h"
#include <future>
// using namespace emp;
using std::future;
using std::cout;
using std::max;
using std::cerr;
using std::endl;
using std::flush;

const static block inProdTableBlock[] = {zero_block, all_one_block};

block inProd(bool * b, block * blk, int length) {
		block res = zero_block;
		for(int i = 0; i < length; ++i) 
//			if(b[i])
//				res = res ^ blk[i];
			res = res ^ (inProdTableBlock[b[i]] & blk[i]);
		return res;
}
#ifdef __GNUC__
	#ifndef __clang__
		#pragma GCC push_options
		#pragma GCC optimize ("unroll-loops")
	#endif
#endif

template<int ssp>
void inProdhelp(block *Ms,  bool * tmp[ssp],  block * MAC, int i) {
	for(int j = 0; j < ssp; ++j)
		Ms[j] = Ms[j] ^ (inProdTableBlock[tmp[j][i]] & MAC[i]);
}
#ifdef __GNUC__
	#ifndef __clang__
		#pragma GCC pop_options
	#endif
#endif


template<int ssp>
void inProds(block *Ms,  bool * tmp[ssp], block * MAC, int length) {
	memset(Ms, 0, sizeof(block)*ssp);
	for(int i = 0; i < length; ++i)  {
		inProdhelp<ssp>(Ms, tmp, MAC, i);
	}
}
bool inProd(bool * b, bool* b2, int length) {
		bool res = false;
		for(int i = 0; i < length; ++i)
			res = (res != (b[i] and b2[i]));
		return res;
}

template<typename T>
void joinNclean(vector<future<T>>& res) {
	for(auto &v: res) v.get();
	res.clear();
}

bool joinNcleanCheat(vector<future<bool>>& res) {
	bool cheat = false;
	for(auto &v: res) cheat = cheat or v.get();
	res.clear();
	return cheat;
}

void send_bool(NetIO * io, const bool * data, int length) {
	if(lan_network) {
		io->send_data(data, length);
		return;
	}
	for(int i = 0; i < length;) {
		uint64_t tmp = 0;
		for(int j = 0; j < 64 and i < length; ++i,++j) {
			if(data[i])
				tmp|=(0x1ULL<<j);
		}
		io->send_data(&tmp, 8);
	}
}

void recv_bool(NetIO * io, bool * data, int length) {
	if(lan_network) {
		io->recv_data(data, length);
		return;
	}
	for(int i = 0; i < length;) {
		uint64_t tmp = 0;
		io->recv_data(&tmp, 8);
		for(int j = 63; j >= 0 and i < length; ++i,--j) {
			data[i] = (tmp&0x1) == 0x1;
			tmp>>=1;
		}
	}
}

template<int B>
void send_partial_block(NetIO * io, const block * data, int length) {
	for(int i = 0; i < length; ++i) {
		io->send_data(&(data[i]), B);
	}
}

template<int B>
void recv_partial_block(NetIO * io, block * data, int length) {
	for(int i = 0; i < length; ++i) {
		io->recv_data(&(data[i]), B);
	}
}

inline uint8_t LSB(block & b) {
	return _mm_extract_epi8(b, 0) & 0x1;
}

template<int nP>
block sampleRandom(NetIOMP<nP> * io, PRG * prg, ThreadPool * pool, int party) {
	vector<future<void>> res;
	vector<future<bool>> res2;
	char (*dgst)[Hash::DIGEST_SIZE] = new char[nP+1][Hash::DIGEST_SIZE];
	block *S = new block[nP+1];
	prg->random_block(&S[party], 1);
	Hash::hash_once(dgst[party], &S[party], sizeof(block));

	for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
		int party2 = i + j - party;
		res.push_back(pool->enqueue([dgst, io, party, party2]() {
			io->send_data(party2, dgst[party], Hash::DIGEST_SIZE);
			io->recv_data(party2, dgst[party2], Hash::DIGEST_SIZE);
		}));
	}
	joinNclean(res);
	for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
		int party2 = i + j - party;
		res2.push_back(pool->enqueue([io, S, dgst, party, party2]() -> bool {
			io->send_data(party2, &S[party], sizeof(block));
			io->recv_data(party2, &S[party2], sizeof(block));
			char tmp[Hash::DIGEST_SIZE];
			Hash::hash_once(tmp, &S[party2], sizeof(block));
			return strncmp(tmp, dgst[party2], Hash::DIGEST_SIZE)!=0;
		}));
	}
	bool cheat = joinNcleanCheat(res2);
	if(cheat) {
		cout <<"cheat in sampleRandom\n"<<flush;
		exit(0);
	}
	for(int i = 2; i <= nP; ++i)
		S[1] = S[1] ^ S[i];
	block result = S[1];
	delete[] S;
	delete[] dgst;
	return result;
}

template<int nP>
void check_MAC(NetIOMP<nP> * io, block * MAC[nP+1], block * KEY[nP+1], bool * r, block Delta, int length, int party) {
	block * tmp = new block[length];
	block tD;
	for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if (i < j) {
		if(party == i) {
			io->send_data(j, &Delta, sizeof(block));
			io->send_data(j, KEY[j], sizeof(block)*length);
			io->flush(j);
		} else if(party == j) {
			io->recv_data(i, &tD, sizeof(block));
			io->recv_data(i, tmp, sizeof(block)*length);
			for(int k = 0; k < length; ++k) {
				if(r[k])tmp[k] = tmp[k] ^ tD;
			}
			if(!cmpBlock(MAC[i], tmp, length))
				error("check_MAC failed!");
		}
	}
	delete[] tmp;
	if(party == 1)
		cerr<<"check_MAC pass!\n"<<flush; 
}

template<int nP>
void check_correctness(NetIOMP<nP>* io, bool * r, int length, int party) {
	if (party == 1) {
		bool * tmp1 = new bool[length*3];
		bool * tmp2 = new bool[length*3];
		memcpy(tmp1, r, length*3);
		for(int i = 2; i <= nP; ++i) {
			io->recv_data(i, tmp2, length*3);
			for(int k = 0; k < length*3; ++k)
				tmp1[k] = (tmp1[k] != tmp2[k]);
		}
		for(int k = 0; k < length; ++k) {
			if((tmp1[3*k] and tmp1[3*k+1]) != tmp1[3*k+2])
				error("check_correctness failed!");
		}
		delete[] tmp1;
		delete[] tmp2;
		cerr<<"check_correctness pass!\n"<<flush; 
	} else {
		io->send_data(1, r, length*3);
		io->flush(1);
	}
}

void print128_num(__m128i *var)
{
    int64_t *v64val = (int64_t*) var;
    printf("%.16lx%.16lx \n", v64val[1], v64val[0]);
}

inline const char* hex_char_to_bin(char c) {
	switch(toupper(c)) {
		case '0': return "0000";
		case '1': return "0001";
		case '2': return "0010";
		case '3': return "0011";
		case '4': return "0100";
		case '5': return "0101";
		case '6': return "0110";
		case '7': return "0111";
		case '8': return "1000";
		case '9': return "1001";
		case 'A': return "1010";
		case 'B': return "1011";
		case 'C': return "1100";
		case 'D': return "1101";
		case 'E': return "1110";
		case 'F': return "1111";
		default: return "0";
	}
}


inline std::string hex_to_binary(std::string hex) {
	std::string bin;
	for(unsigned i = 0; i != hex.length(); ++i)
		bin += hex_char_to_bin(hex[i]);
	return bin;
}


#define SIGN_BIT 0x80000000
#define EXP_BIT  0x7f800000
#define TAIL_BIT 0x007fffff

int64_t Float2FPA(float* aFloat) {

		int64_t signal = 0;
        int64_t aFix = 0;//定点数容器
        int tmp = 0;//浮点数容器
        int exp = 0;//指数大小
        int tail = 0;//尾数位容器
        tmp = *((int*)aFloat);//置定点数的符号位 
        signal = (tmp & SIGN_BIT) >> 31;

        //置定点数的整数部分
        exp = ((tmp & EXP_BIT) >> 23) - 127;//指数值
        tail = ((tmp & TAIL_BIT) | 0x00800000);//尾数各位 
        aFix = aFix | ((tail >> (23-exp)) << 13);

        //置定点数的小数部分
		//判断小数部分是否大于13bit
		if ((23-exp) > 13)
			aFix = aFix | ((tail & ~(0xffffffff << (23-exp))) >> (23-exp-13));
		else
			aFix = aFix | ((tail & ~(0xffffffff << (23-exp))) << (13-23+exp));
		
		if (signal)
		{
			aFix = (~aFix) + 1;
		}
		return aFix;
}

// float FPA2Float(int64_t aFix) {


// }

#endif// __HELPER

