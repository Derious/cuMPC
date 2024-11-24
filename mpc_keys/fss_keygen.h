#ifndef FSS_KEYGEN_H
#define FSS_KEYGEN_H

#include "aes_prg_host.h"
#include <time.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

// select vector table from 0 to 127 
extern const uint128_t select_vector_table[128] = {
    uint128_t(0x8000000000000000ULL, 0x0000000000000000ULL), // 0
    uint128_t(0x4000000000000000ULL, 0x0000000000000000ULL), // 1
    uint128_t(0x2000000000000000ULL, 0x0000000000000000ULL), // 2
    uint128_t(0x1000000000000000ULL, 0x0000000000000000ULL), // 3
    uint128_t(0x0800000000000000ULL, 0x0000000000000000ULL), // 4
    uint128_t(0x0400000000000000ULL, 0x0000000000000000ULL), // 5
    uint128_t(0x0200000000000000ULL, 0x0000000000000000ULL), // 6
    uint128_t(0x0100000000000000ULL, 0x0000000000000000ULL), // 7
    uint128_t(0x0080000000000000ULL, 0x0000000000000000ULL), // 8
    uint128_t(0x0040000000000000ULL, 0x0000000000000000ULL), // 9
    uint128_t(0x0020000000000000ULL, 0x0000000000000000ULL), // 10
    uint128_t(0x0010000000000000ULL, 0x0000000000000000ULL), // 11
    uint128_t(0x0008000000000000ULL, 0x0000000000000000ULL), // 12
    uint128_t(0x0004000000000000ULL, 0x0000000000000000ULL), // 13
    uint128_t(0x0002000000000000ULL, 0x0000000000000000ULL), // 14
    uint128_t(0x0001000000000000ULL, 0x0000000000000000ULL), // 15
    uint128_t(0x0000800000000000ULL, 0x0000000000000000ULL), // 16
    uint128_t(0x0000400000000000ULL, 0x0000000000000000ULL), // 17
    uint128_t(0x0000200000000000ULL, 0x0000000000000000ULL), // 18
    uint128_t(0x0000100000000000ULL, 0x0000000000000000ULL), // 19
    uint128_t(0x0000080000000000ULL, 0x0000000000000000ULL), // 20
    uint128_t(0x0000040000000000ULL, 0x0000000000000000ULL), // 21
    uint128_t(0x0000020000000000ULL, 0x0000000000000000ULL), // 22
    uint128_t(0x0000010000000000ULL, 0x0000000000000000ULL), // 23
    uint128_t(0x0000008000000000ULL, 0x0000000000000000ULL), // 24
    uint128_t(0x0000004000000000ULL, 0x0000000000000000ULL), // 25
    uint128_t(0x0000002000000000ULL, 0x0000000000000000ULL), // 26
    uint128_t(0x0000001000000000ULL, 0x0000000000000000ULL), // 27
    uint128_t(0x0000000800000000ULL, 0x0000000000000000ULL), // 28
    uint128_t(0x0000000400000000ULL, 0x0000000000000000ULL), // 29
    uint128_t(0x0000000200000000ULL, 0x0000000000000000ULL), // 30
    uint128_t(0x0000000100000000ULL, 0x0000000000000000ULL), // 31
    uint128_t(0x0000000080000000ULL, 0x0000000000000000ULL), // 32
    uint128_t(0x0000000040000000ULL, 0x0000000000000000ULL), // 33
    uint128_t(0x0000000020000000ULL, 0x0000000000000000ULL), // 34
    uint128_t(0x0000000010000000ULL, 0x0000000000000000ULL), // 35
    uint128_t(0x0000000008000000ULL, 0x0000000000000000ULL), // 36
    uint128_t(0x0000000004000000ULL, 0x0000000000000000ULL), // 37
    uint128_t(0x0000000002000000ULL, 0x0000000000000000ULL), // 38
    uint128_t(0x0000000001000000ULL, 0x0000000000000000ULL), // 39
    uint128_t(0x0000000000800000ULL, 0x0000000000000000ULL), // 40
    uint128_t(0x0000000000400000ULL, 0x0000000000000000ULL), // 41
    uint128_t(0x0000000000200000ULL, 0x0000000000000000ULL), // 42
    uint128_t(0x0000000000100000ULL, 0x0000000000000000ULL), // 43
    uint128_t(0x0000000000080000ULL, 0x0000000000000000ULL), // 44
    uint128_t(0x0000000000040000ULL, 0x0000000000000000ULL), // 45
    uint128_t(0x0000000000020000ULL, 0x0000000000000000ULL), // 46
    uint128_t(0x0000000000010000ULL, 0x0000000000000000ULL), // 47
    uint128_t(0x0000000000008000ULL, 0x0000000000000000ULL), // 48
    uint128_t(0x0000000000004000ULL, 0x0000000000000000ULL), // 49
    uint128_t(0x0000000000002000ULL, 0x0000000000000000ULL), // 50
    uint128_t(0x0000000000001000ULL, 0x0000000000000000ULL), // 51
    uint128_t(0x0000000000000800ULL, 0x0000000000000000ULL), // 52
    uint128_t(0x0000000000000400ULL, 0x0000000000000000ULL), // 53
    uint128_t(0x0000000000000200ULL, 0x0000000000000000ULL), // 54
    uint128_t(0x0000000000000100ULL, 0x0000000000000000ULL), // 55
    uint128_t(0x0000000000000080ULL, 0x0000000000000000ULL), // 56
    uint128_t(0x0000000000000040ULL, 0x0000000000000000ULL), // 57
    uint128_t(0x0000000000000020ULL, 0x0000000000000000ULL), // 58
    uint128_t(0x0000000000000010ULL, 0x0000000000000000ULL), // 59
    uint128_t(0x0000000000000008ULL, 0x0000000000000000ULL), // 60
    uint128_t(0x0000000000000004ULL, 0x0000000000000000ULL), // 61
    uint128_t(0x0000000000000002ULL, 0x0000000000000000ULL), // 62
    uint128_t(0x0000000000000001ULL, 0x0000000000000000ULL), // 63
    uint128_t(0x0000000000000000ULL, 0x8000000000000000ULL), // 64
    uint128_t(0x0000000000000000ULL, 0x4000000000000000ULL), // 65
    uint128_t(0x0000000000000000ULL, 0x2000000000000000ULL), // 66
    uint128_t(0x0000000000000000ULL, 0x1000000000000000ULL), // 67
    uint128_t(0x0000000000000000ULL, 0x0800000000000000ULL), // 68
    uint128_t(0x0000000000000000ULL, 0x0400000000000000ULL), // 69
    uint128_t(0x0000000000000000ULL, 0x0200000000000000ULL), // 70
    uint128_t(0x0000000000000000ULL, 0x0100000000000000ULL), // 71
    uint128_t(0x0000000000000000ULL, 0x0080000000000000ULL), // 72
    uint128_t(0x0000000000000000ULL, 0x0040000000000000ULL), // 73
    uint128_t(0x0000000000000000ULL, 0x0020000000000000ULL), // 74
    uint128_t(0x0000000000000000ULL, 0x0010000000000000ULL), // 75
    uint128_t(0x0000000000000000ULL, 0x0008000000000000ULL), // 76
    uint128_t(0x0000000000000000ULL, 0x0004000000000000ULL), // 77
    uint128_t(0x0000000000000000ULL, 0x0002000000000000ULL), // 78
    uint128_t(0x0000000000000000ULL, 0x0001000000000000ULL), // 79
    uint128_t(0x0000000000000000ULL, 0x0000800000000000ULL), // 80
    uint128_t(0x0000000000000000ULL, 0x0000400000000000ULL), // 81
    uint128_t(0x0000000000000000ULL, 0x0000200000000000ULL), // 82
    uint128_t(0x0000000000000000ULL, 0x0000100000000000ULL), // 83
    uint128_t(0x0000000000000000ULL, 0x0000080000000000ULL), // 84
    uint128_t(0x0000000000000000ULL, 0x0000040000000000ULL), // 85
    uint128_t(0x0000000000000000ULL, 0x0000020000000000ULL), // 86
    uint128_t(0x0000000000000000ULL, 0x0000010000000000ULL), // 87
    uint128_t(0x0000000000000000ULL, 0x0000008000000000ULL), // 88
    uint128_t(0x0000000000000000ULL, 0x0000004000000000ULL), // 89
    uint128_t(0x0000000000000000ULL, 0x0000002000000000ULL), // 90
    uint128_t(0x0000000000000000ULL, 0x0000001000000000ULL), // 91
    uint128_t(0x0000000000000000ULL, 0x0000000800000000ULL), // 92
    uint128_t(0x0000000000000000ULL, 0x0000000400000000ULL), // 93
    uint128_t(0x0000000000000000ULL, 0x0000000200000000ULL), // 94
    uint128_t(0x0000000000000000ULL, 0x0000000100000000ULL), // 95
    uint128_t(0x0000000000000000ULL, 0x0000000080000000ULL), // 96
    uint128_t(0x0000000000000000ULL, 0x0000000040000000ULL), // 97
    uint128_t(0x0000000000000000ULL, 0x0000000020000000ULL), // 98
    uint128_t(0x0000000000000000ULL, 0x0000000010000000ULL), // 99
    uint128_t(0x0000000000000000ULL, 0x0000000008000000ULL), // 100
    uint128_t(0x0000000000000000ULL, 0x0000000004000000ULL), // 101
    uint128_t(0x0000000000000000ULL, 0x0000000002000000ULL), // 102
    uint128_t(0x0000000000000000ULL, 0x0000000001000000ULL), // 103
    uint128_t(0x0000000000000000ULL, 0x0000000000800000ULL), // 104
    uint128_t(0x0000000000000000ULL, 0x0000000000400000ULL), // 105
    uint128_t(0x0000000000000000ULL, 0x0000000000200000ULL), // 106
    uint128_t(0x0000000000000000ULL, 0x0000000000100000ULL), // 107
    uint128_t(0x0000000000000000ULL, 0x0000000000080000ULL), // 108
    uint128_t(0x0000000000000000ULL, 0x0000000000040000ULL), // 109
    uint128_t(0x0000000000000000ULL, 0x0000000000020000ULL), // 110
    uint128_t(0x0000000000000000ULL, 0x0000000000010000ULL), // 111
    uint128_t(0x0000000000000000ULL, 0x0000000000008000ULL), // 112
    uint128_t(0x0000000000000000ULL, 0x0000000000004000ULL), // 113
    uint128_t(0x0000000000000000ULL, 0x0000000000002000ULL), // 114
    uint128_t(0x0000000000000000ULL, 0x0000000000001000ULL), // 115
    uint128_t(0x0000000000000000ULL, 0x0000000000000800ULL), // 116
    uint128_t(0x0000000000000000ULL, 0x0000000000000400ULL), // 117
    uint128_t(0x0000000000000000ULL, 0x0000000000000200ULL), // 118
    uint128_t(0x0000000000000000ULL, 0x0000000000000100ULL), // 119
    uint128_t(0x0000000000000000ULL, 0x0000000000000080ULL), // 120
    uint128_t(0x0000000000000000ULL, 0x0000000000000040ULL), // 121
    uint128_t(0x0000000000000000ULL, 0x0000000000000020ULL), // 122
    uint128_t(0x0000000000000000ULL, 0x0000000000000010ULL), // 123
    uint128_t(0x0000000000000000ULL, 0x0000000000000008ULL), // 124
    uint128_t(0x0000000000000000ULL, 0x0000000000000004ULL), // 125
    uint128_t(0x0000000000000000ULL, 0x0000000000000002ULL), // 126
    uint128_t(0x0000000000000000ULL, 0x0000000000000001ULL) // 127
};

void Double_PRG(AES_KEY *key, uint128_t input, uint128_t& output1, uint128_t& output2, int& bit1, int& bit2){
	input = input.set_lsb_zero();

	uint128_t stash[2];
	stash[0] = input;
	stash[1] = input.reverse_lsb();

	block stash_block[2];
	stash_block[0] = makeBlock(stash[0].get_high(), stash[0].get_low());
	stash_block[1] = makeBlock(stash[1].get_high(), stash[1].get_low());

	AES_ecb_encrypt_blks(stash_block, 2, key);
	int64_t *v64val = (int64_t*) &stash_block[0];
	stash[0] = uint128_t(v64val[1], v64val[0]);
	v64val = (int64_t*) &stash_block[1];
	stash[1] = uint128_t(v64val[1], v64val[0]);


	stash[0] = stash[0] ^ input;
	stash[1] = stash[1] ^ input;
	stash[1] = stash[1].reverse_lsb();

	bit1 = stash[0].get_lsb();
	bit2 = stash[1].get_lsb();

	output1 = stash[0].set_lsb_zero();
	output2 = stash[1].set_lsb_zero();
}

void GEN_Pack(AES_Generator* prg, AES_KEY *key, uint128_t alpha, int n, unsigned char** k0, unsigned char **k1){
	int maxlayer = n - 7;
	//int maxlayer = n;

	uint128_t s[maxlayer + 1][2];
	int t[maxlayer + 1 ][2];
	uint128_t sCW[maxlayer];
	int tCW[maxlayer][2];

	s[0][0] = prg->random(); 
	s[0][1] = prg->random();
    s[0][0].print_uint128("s[0][0] = ", s[0][0]);
    s[0][1].print_uint128("s[0][1] = ", s[0][1]);
	t[0][0] = s[0][0].get_lsb();
	t[0][1] = t[0][0] ^ 1;
	s[0][0] = s[0][0].set_lsb_zero();
	s[0][1] = s[0][1].set_lsb_zero();

	int i;
	uint128_t s0[2], s1[2]; // 0=L,1=R
	#define LEFT 0
	#define RIGHT 1
	int t0[2], t1[2];
	for(i = 1; i<= maxlayer; i++){
		Double_PRG(key, s[i-1][0], s0[LEFT], s0[RIGHT], t0[LEFT], t0[RIGHT]);
		Double_PRG(key, s[i-1][1], s1[LEFT], s1[RIGHT], t1[LEFT], t1[RIGHT]);

		int keep, lose;
        // int alphabit = getbit(alpha, n, i);
        int alphabit = alpha.get_bit(n-i);
		if(alphabit == 0){
			keep = LEFT;
			lose = RIGHT;
		}else{
			keep = RIGHT;
			lose = LEFT;
		}

		sCW[i-1] = s0[lose] ^ s1[lose];

		tCW[i-1][LEFT] = t0[LEFT] ^ t1[LEFT] ^ alphabit ^ 1;
		tCW[i-1][RIGHT] = t0[RIGHT] ^ t1[RIGHT] ^ alphabit;

		if(t[i-1][0] == 1){
			s[i][0] = s0[keep] ^ sCW[i-1];
			t[i][0] = t0[keep] ^ tCW[i-1][keep];
		}else{
			s[i][0] = s0[keep];
			t[i][0] = t0[keep];
		}

		if(t[i-1][1] == 1){
			s[i][1] = s1[keep] ^ sCW[i-1];
			t[i][1] = t1[keep] ^ tCW[i-1][keep];
		}else{
			s[i][1] = s1[keep];
			t[i][1] = t1[keep];
		}
	}

    uint128_t finalblock;
    finalblock = select_vector_table[alpha.get_low() & 127];
    finalblock.print_uint128("select_vector = ", finalblock);
    s[maxlayer][0].print_uint128("s[maxlayer][0] = ", s[maxlayer][0]);
    s[maxlayer][1].print_uint128("s[maxlayer][1] = ", s[maxlayer][1]);
	finalblock = finalblock ^ s[maxlayer][0];
	finalblock = finalblock ^ s[maxlayer][1];
    finalblock.print_uint128("finalblock = ", finalblock);

	unsigned char *buff0;
	unsigned char *buff1;
	buff0 = (unsigned char*) malloc(1 + 16 + 1 + 18 * maxlayer + 16);
	buff1 = (unsigned char*) malloc(1 + 16 + 1 + 18 * maxlayer + 16);

	if(buff0 == NULL || buff1 == NULL){
		printf("Memory allocation failed\n");
		exit(1);
	}

	buff0[0] = n;
	memcpy(&buff0[1], &s[0][0], 16);
	buff0[17] = t[0][0];
	for(i = 1; i <= maxlayer; i++){
		memcpy(&buff0[18 * i], &sCW[i-1], 16);
		buff0[18 * i + 16] = tCW[i-1][0];
		buff0[18 * i + 17] = tCW[i-1][1]; 
	}
	memcpy(&buff0[18 * maxlayer + 18], &finalblock, 16); 

	buff1[0] = n;
	memcpy(&buff1[18], &buff0[18], 18 * (maxlayer));
	memcpy(&buff1[1], &s[0][1], 16);
	buff1[17] = t[0][1];
	memcpy(&buff1[18 * maxlayer + 18], &finalblock, 16);

	*k0 = buff0;
	*k1 = buff1;
}

uint128_t EVAL_Pack(AES_KEY *key, unsigned char* k, uint128_t x){
	int n = k[0];
	int maxlayer = n - 7;

	uint128_t s[maxlayer + 1];
	int t[maxlayer + 1];
	uint128_t sCW[maxlayer];
	int tCW[maxlayer][2];
	uint128_t finalblock;

	memcpy(&s[0], &k[1], 16);
	t[0] = k[17];

	int i;
	for(i = 1; i <= maxlayer; i++){
		memcpy(&sCW[i-1], &k[18 * i], 16);
		tCW[i-1][0] = k[18 * i + 16];
		tCW[i-1][1] = k[18 * i + 17];
	}

	memcpy(&finalblock, &k[18 * (maxlayer + 1)], 16);

	uint128_t sL, sR;
	int tL, tR;
	for(i = 1; i <= maxlayer; i++){
		Double_PRG(key, s[i - 1], sL, sR, tL, tR); 

		if(t[i-1] == 1){
			sL = sL ^ sCW[i-1];
			sR = sR ^ sCW[i-1];
			tL = tL ^ tCW[i-1][0];
			tR = tR ^ tCW[i-1][1];	
		}

		int xbit = x.get_bit(n-i);
		if(xbit == 0){
			s[i] = sL;
			t[i] = tL;
		}else{
			s[i] = sR;
			t[i] = tR;
		}
	}

	uint128_t res;
	res = s[maxlayer];
    res = res ^ finalblock.select(t[maxlayer]);
	return res;
}

void fss_gen(AES_Generator* prg, AES_KEY *key, uint128_t alpha, int n, unsigned char** k0, unsigned char **k1){
	// int maxlayer = n - 7;
	int maxlayer = n;

	uint128_t s[maxlayer + 1][2];
	int t[maxlayer + 1 ][2];
	uint128_t sCW[maxlayer];
	int tCW[maxlayer][2];

	s[0][0] = prg->random(); 
	s[0][1] = prg->random();
	t[0][0] = s[0][0].get_lsb();
	t[0][1] = t[0][0] ^ 1;
	s[0][0] = s[0][0].set_lsb_zero();
	s[0][1] = s[0][1].set_lsb_zero();

	int i;
	uint128_t s0[2], s1[2]; // 0=L,1=R
	#define LEFT 0
	#define RIGHT 1
	int t0[2], t1[2];
	for(i = 1; i<= maxlayer; i++){
		Double_PRG(key, s[i-1][0], s0[LEFT], s0[RIGHT], t0[LEFT], t0[RIGHT]);
		Double_PRG(key, s[i-1][1], s1[LEFT], s1[RIGHT], t1[LEFT], t1[RIGHT]);

		int keep, lose;
        // int alphabit = getbit(alpha, n, i);
        int alphabit = alpha.get_bit(n-i);
		if(alphabit == 0){
			keep = LEFT;
			lose = RIGHT;
		}else{
			keep = RIGHT;
			lose = LEFT;
		}

		sCW[i-1] = s0[lose] ^ s1[lose];

		tCW[i-1][LEFT] = t0[LEFT] ^ t1[LEFT] ^ alphabit ^ 1;
		tCW[i-1][RIGHT] = t0[RIGHT] ^ t1[RIGHT] ^ alphabit;

		if(t[i-1][0] == 1){
			s[i][0] = s0[keep] ^ sCW[i-1];
			t[i][0] = t0[keep] ^ tCW[i-1][keep];
		}else{
			s[i][0] = s0[keep];
			t[i][0] = t0[keep];
		}

		if(t[i-1][1] == 1){
			s[i][1] = s1[keep] ^ sCW[i-1];
			t[i][1] = t1[keep] ^ tCW[i-1][keep];
		}else{
			s[i][1] = s1[keep];
			t[i][1] = t1[keep];
		}
	}

    uint128_t finalblock(0,1);
	finalblock = finalblock ^ s[maxlayer][0];
	finalblock = finalblock ^ s[maxlayer][1];
    // finalblock.print_uint128("finalblock = ", finalblock);

	unsigned char *buff0;
	unsigned char *buff1;
	buff0 = (unsigned char*) malloc(1 + 16 + 1 + 18 * maxlayer + 16);
	buff1 = (unsigned char*) malloc(1 + 16 + 1 + 18 * maxlayer + 16);

	if(buff0 == NULL || buff1 == NULL){
		printf("Memory allocation failed\n");
		exit(1);
	}

	buff0[0] = n;
	memcpy(&buff0[1], &s[0][0], 16);
	buff0[17] = t[0][0];
	for(i = 1; i <= maxlayer; i++){
		memcpy(&buff0[18 * i], &sCW[i-1], 16);
		buff0[18 * i + 16] = tCW[i-1][0];
		buff0[18 * i + 17] = tCW[i-1][1]; 
	}
	memcpy(&buff0[18 * maxlayer + 18], &finalblock, 16); 

	buff1[0] = n;
	memcpy(&buff1[18], &buff0[18], 18 * (maxlayer));
	memcpy(&buff1[1], &s[0][1], 16);
	buff1[17] = t[0][1];
	memcpy(&buff1[18 * maxlayer + 18], &finalblock, 16);

	*k0 = buff0;
	*k1 = buff1;
}

uint128_t dcf_eval(AES_KEY *key, unsigned char* k, uint128_t x){
	int n = k[0];
	int maxlayer = n;

	uint128_t s[maxlayer + 1];
	int t[maxlayer + 1];
	uint128_t sCW[maxlayer];
	int tCW[maxlayer][2];
	uint128_t finalblock;

	memcpy(&s[0], &k[1], 16);
	t[0] = k[17];

	int i;
	for(i = 1; i <= maxlayer; i++){
		memcpy(&sCW[i-1], &k[18 * i], 16);
		tCW[i-1][0] = k[18 * i + 16];
		tCW[i-1][1] = k[18 * i + 17];
	}

	memcpy(&finalblock, &k[18 * (maxlayer + 1)], 16);

	uint128_t sL, sR;
	uint128_t res(0,0);
	int tL, tR;

    // first layer
    Double_PRG(key, s[0], sL, sR, tL, tR); 

	sL = sL ^ sCW[0].select(t[0]);
	sR = sR ^ sCW[0].select(t[0]);
	tL = tL ^ (tCW[0][0]*t[0]);
	tR = tR ^ (tCW[0][1]*t[0]);	

	int xbit = x.get_bit(n-1);

    s[1] = sR.select(xbit) ^ sL.select((1-xbit));
    t[1] = tR * xbit + tL * (1-xbit);

    res = res ^ uint128_t(0, xbit*t[0]);

	for(i = 2; i <= maxlayer; i++){
        Double_PRG(key, s[i - 1], sL, sR, tL, tR); 

		sL = sL ^ sCW[i-1].select(t[i-1]);
		sR = sR ^ sCW[i-1].select(t[i-1]);
		tL = tL ^ (tCW[i-1][0]*t[i-1]);
		tR = tR ^ (tCW[i-1][1]*t[i-1]);	

		int xbit = x.get_bit(n-i);
		s[i] = sR.select(xbit) ^ sL.select((1-xbit));
		t[i] = tR * xbit + tL * (1-xbit);

        int xbit_last = x.get_bit(n-i+1);
        int changed = (xbit_last * (1 - xbit)) | ((1 - xbit_last) * xbit);
        res = res ^ uint128_t(0, changed*t[i-1]);
	}
	return res;
}

#endif // FSS_KEYGEN_H