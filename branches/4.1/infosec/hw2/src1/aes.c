#include <stdio.h>
#include <string.h>
#include <malloc.h>

#include "aes.h"
#include "types.h"

const ubyte_t sbox[256] = {
    //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, //0
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, //1
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, //2
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, //3
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, //4
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, //5
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, //6
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, //7
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, //8
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, //9
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, //A
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, //B
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, //C
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, //D
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, //E
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16	//F
};

int inv_sbox[256] = {
    //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

const ubyte_t Rcon[] = { 0x01, 0x02, 0x04, 0x08,
				   0x10, 0x20, 0x40, 0x80,
				   0x1B, 0x36, 0x6c, 0xD8 };

unsigned char xtime(unsigned char a, unsigned char b)
{
	unsigned char temp;
	if (b == 0x00)
		return 0x00;
	if (b == 0x01)
		return a;

	temp = xtime(a, b >> 1);
	return (temp << 1) ^ ((temp >> 7) * 0x1b);
}

ubyte_t GF_Mul_X2(ubyte_t a)
{
	return xtime(a, 0x02);
}

ubyte_t GF_Mul_X3(ubyte_t a)
{
	return xtime(a, 0x02) ^ a;
}

ubyte_t GF_Mul_X9(ubyte_t a)
{
	return xtime(a, 0x08) ^ xtime(a, 0x01);
}

ubyte_t GF_Mul_Xb(ubyte_t a)
{
	return xtime(a, 0x08) ^ xtime(a, 0x02) ^ xtime(a, 0x01);
}

ubyte_t GF_Mul_Xd(ubyte_t a)
{
	return xtime(a, 0x08) ^ xtime(a, 0x04) ^ xtime(a, 0x01);
}

ubyte_t GF_Mul_Xe(ubyte_t a)
{
	return xtime(a, 0x08) ^ xtime(a, 0x04) ^ xtime(a, 0x02);
}

void SubBytes(unsigned char * state, int length)
{
	int i;

	for (i = 0; i < length; ++i) {
		state[i] = sbox[state[i]];
	}
}

void InvSubBytes(unsigned char * state, int length)
{
	int i;

	for (i = 0; i < length; ++i) {
		state[i] = inv_sbox[state[i]];
	}
}

void ShiftRows(unsigned char * state, int length)
{
	unsigned char * t;
	int Nb;
	int base;
	int i, j;

	t = (unsigned char *) malloc(sizeof(unsigned char) * length);
	memcpy(t, state, sizeof(unsigned char) * length);

	Nb = length >> 2;

	for (i = 0; i < Nb; ++i) {
		base = i << 2;
		for (j = 1; j < ROW_COUNT; ++j) {
			state[base+j] = t[((i+j)%Nb)*4 + j];
		}
	}
	free(t);
}

void InvShiftRows(unsigned char * state, int length)
{
	unsigned char * t;
	int Nb;
	int base;
	int i, j;

	t = (unsigned char *) malloc(sizeof(unsigned char) * length);
	memcpy(t, state, sizeof(unsigned char) * length);

	Nb = length >> 2;

	for (i = 0; i < Nb; ++i) {
		base = i << 2;
		for (j = 1; j < ROW_COUNT; ++j) {
			state[base+j] = t[((i+Nb-j)%Nb)*4 + j];
		}
	}
	free(t);
}

void MixColumns(unsigned char * state, int length)
{
	unsigned char * temp;
	int Nb = length / 4;
	int i;

	temp = (unsigned char *) malloc(sizeof(unsigned char) * ROW_COUNT);
	
	for (i = 0; i < Nb; ++i) {
		int base = i * 4;
		memcpy(temp, state+base, sizeof(unsigned char) * ROW_COUNT);
		
		state[base] = GF_Mul_X2(temp[0]) ^ GF_Mul_X3(temp[1])
			^ temp[2] ^ temp[3];
		state[base+1] = temp[0] ^ GF_Mul_X2(temp[1])
			^ GF_Mul_X3(temp[2]) ^ temp[3];
		state[base+2] = temp[0] ^ temp[1]
			^ GF_Mul_X2(temp[2]) ^ GF_Mul_X3(temp[3]);
		state[base+3] = GF_Mul_X3(temp[0]) ^ temp[1]
			^ temp[2] ^ GF_Mul_X2(temp[3]);
	}
	free(temp);
}

void InvMixColumns(unsigned char * state, int length)
{
	unsigned char * temp;
	int Nb = length / 4;
	int i;

	temp = (unsigned char *) malloc(sizeof(unsigned char) * ROW_COUNT);
	
	for (i = 0; i < Nb; ++i) {
		int base = i * 4;
		memcpy(temp, state+base, sizeof(unsigned char) * ROW_COUNT);
		
		state[base] = GF_Mul_Xe(temp[0]) ^ GF_Mul_Xb(temp[1])
			^ GF_Mul_Xd(temp[2]) ^ GF_Mul_X9(temp[3]);
		state[base+1] = GF_Mul_X9(temp[0]) ^ GF_Mul_Xe(temp[1])
			^ GF_Mul_Xb(temp[2]) ^ GF_Mul_Xd(temp[3]);
		state[base+2] = GF_Mul_Xd(temp[0]) ^ GF_Mul_X9(temp[1])
			^ GF_Mul_Xe(temp[2]) ^ GF_Mul_Xb(temp[3]);
		state[base+3] = GF_Mul_Xb(temp[0]) ^ GF_Mul_Xd(temp[1])
			^ GF_Mul_X9(temp[2]) ^ GF_Mul_Xe(temp[3]);
		/*state[base+1] = temp[0] ^ GF_Mul_X2(temp[1])*/
			/*^ GF_Mul_X3(temp[2]) ^ temp[3];*/
		/*state[base+2] = temp[0] ^ temp[1]*/
			/*^ GF_Mul_X2(temp[2]) ^ GF_Mul_X3(temp[3]);*/
		/*state[base+3] = GF_Mul_X3(temp[0]) ^ temp[1]*/
			/*^ temp[2] ^ GF_Mul_X2(temp[3]);*/
	}
	free(temp);
}

void AddRoundKey(ubyte_t * state, ubyte_t * key, int length)
{
	int i;

	for (i = 0; i < length; ++i) {
		state[i] = state[i] ^ key[i];
	}
}

// key: 0,Nk  key_pool:0,Nb(Nr+1)
void KeyExpansion(ubyte_t * key, ubyte_t * key_pool, int Nk, int Nb, int Nr)
{
	uword_t temp;
	ubyte_t *ptemp = (ubyte_t *) &temp;
	uword_t * key_temp = (uword_t *) key_pool;
	int Nek = Nb * (Nr+1);
	int i;
	
	for (i = 0; i < Nk; ++i) {
		int base = i * 4;

		key_pool[base] = key[base];
		key_pool[base+1] = key[base+1];
		key_pool[base+2] = key[base+2];
		key_pool[base+3] = key[base+3];
	}

	for (i = Nk; i < Nek; ++i) {
		//temp = key_temp[i-1];
		int base = i * 4;
		if (i%Nk == 0) {
			ptemp[0] = sbox[key_pool[base-3]] ^ Rcon[i/Nk - 1];
			ptemp[1] = sbox[key_pool[base-2]];
			ptemp[2] = sbox[key_pool[base-1]];
			ptemp[3] = sbox[key_pool[base-4]];

		} else if (Nk > 6 && i%Nk == 4) {
			ptemp[0] = sbox[key_pool[base-4]];
			ptemp[1] = sbox[key_pool[base-3]];
			ptemp[2] = sbox[key_pool[base-2]];
			ptemp[3] = sbox[key_pool[base-1]];

		} else {
			ptemp[0] = key_pool[base-4];
			ptemp[1] = key_pool[base-3];
			ptemp[2] = key_pool[base-2];
			ptemp[3] = key_pool[base-1];
		}
			
		key_temp[i] = key_temp[i-Nk] ^ temp;
		//printf("%02x%02x%02x%02x\n", key_pool[base], key_pool[base+1], key_pool[base+2], key_pool[base+3]);
	}

}

// key: 0,Nk  key_pool:0,Nb(Nr+1)
void InvKeyExpansion(ubyte_t * key, ubyte_t * key_pool, int Nk, int Nb, int Nr)
{
	uword_t temp;
	ubyte_t *ptemp = (ubyte_t *) &temp;
	uword_t * key_temp = (uword_t *) key_pool;
	int Nek = Nb * (Nr+1);
	int i;
	
	for (i = 0; i < Nk; ++i) {
		int base = i * 4;

		key_pool[base] = key[base];
		key_pool[base+1] = key[base+1];
		key_pool[base+2] = key[base+2];
		key_pool[base+3] = key[base+3];
	}

	for (i = Nk; i < Nek; ++i) {
		//temp = key_temp[i-1];
		int base = i * 4;
		if (i%Nk == 0) {
			ptemp[0] = sbox[key_pool[base-3]] ^ Rcon[i/Nk - 1];
			ptemp[1] = sbox[key_pool[base-2]];
			ptemp[2] = sbox[key_pool[base-1]];
			ptemp[3] = sbox[key_pool[base-4]];

		} else if (Nk > 6 && i%Nk == 4) {
			ptemp[0] = sbox[key_pool[base-4]];
			ptemp[1] = sbox[key_pool[base-3]];
			ptemp[2] = sbox[key_pool[base-2]];
			ptemp[3] = sbox[key_pool[base-1]];

		} else {
			ptemp[0] = key_pool[base-4];
			ptemp[1] = key_pool[base-3];
			ptemp[2] = key_pool[base-2];
			ptemp[3] = key_pool[base-1];
		}
			
		key_temp[i] = key_temp[i-Nk] ^ temp;
		//printf("%02x%02x%02x%02x\n", key_pool[base], key_pool[base+1], key_pool[base+2], key_pool[base+3]);
	}

	// addition routine for inverse process
	for (i = 1; i < Nr; ++i) {
		InvMixColumns(key_pool+i*Nb*ROW_COUNT, Nb*ROW_COUNT);
	}
}

void AESEncrypt(ubyte_t * input, ubyte_t * key, ubyte_t * output, int Nb, int Nk, int Nr)
{
	int length_text = Nb * 4;
	ubyte_t * key_pool;
	int i, j;

	memcpy(output, input, sizeof(ubyte_t) * length_text);
	key_pool = (ubyte_t *) malloc(sizeof(ubyte_t)*length_text*(Nr+1));

	KeyExpansion(key, key_pool, Nk, Nb, Nr);

	//0 round
	AddRoundKey(output, key_pool, length_text);
#ifdef DEBUG
		for (j = 0; j < 16; ++j) {
			printf("0x%02x, ", output[j]);
			if (!((j+1) % 4))
				printf("\n");
		}
#endif

	for (i = 1; i <= Nr; ++i) {
		SubBytes(output, length_text);

		ShiftRows(output, length_text);

		if (i != Nr)
			MixColumns(output, length_text);

		AddRoundKey(output, key_pool+i*length_text, length_text);
#ifdef DEBUG
		printf("Round %d\n", i);
		for (j = 0; j < 16; ++j) {
			printf("0x%02x, ", output[j]);
			if (!((j+1) % 4))
				printf("\n");
		}
#endif
	}

	free(key_pool);
}

void AESDEncrypt(ubyte_t * input, ubyte_t * key, ubyte_t * output, int Nb, int Nk, int Nr)
{
	int length_text = Nb * 4;
	ubyte_t * key_pool;
	int i, j;

	memcpy(output, input, sizeof(ubyte_t) * length_text);
	key_pool = (ubyte_t *) malloc(sizeof(ubyte_t)*length_text*(Nr+1));

	InvKeyExpansion(key, key_pool, Nk, Nb, Nr);

	//0 round
	AddRoundKey(output, key_pool+Nr*length_text, length_text);
#ifdef DEBUG
		for (j = 0; j < 16; ++j) {
			printf("0x%02x, ", output[j]);
			if (!((j+1) % 4))
				printf("\n");
		}
#endif

	for (i = Nr-1; i >= 1; --i) {
		InvSubBytes(output, length_text);
		InvShiftRows(output, length_text);
		InvMixColumns(output, length_text);
		AddRoundKey(output, key_pool+i*length_text, length_text);
#ifdef DEBUG
		printf("Round %d\n", i);
		for (j = 0; j < 16; ++j) {
			printf("0x%02x, ", output[j]);
			if (!((j+1) % 4))
				printf("\n");
		}
#endif
	}

	//Last round
	InvSubBytes(output, length_text);
	InvShiftRows(output, length_text);
	AddRoundKey(output, key_pool, length_text);

	free(key_pool);
}

int main()
{
	//original
	unsigned char plain[] = { 0x32, 0x43, 0xf6, 0xa8,
							  0x88, 0x5a, 0x30, 0x8d,
							  0x31, 0x31, 0x98, 0xa2,
							  0xe0, 0x37, 0x07, 0x34};
//at the beginning of round 1
	/*unsigned char plain[] = { 0x19, 0x3d, 0xe3, 0xbe,*/
							  /*0xa0, 0xf4, 0xe2, 0x2b,*/
							  /*0x9a, 0xc6, 0x8d, 0x2a,*/
							  /*0xe9, 0xf8, 0x48, 0x08};*/
//	after subbytes
	/*unsigned char plain[] = { 0xd4, 0x27, 0x11, 0xae, */
							  /*0xe0, 0xbf, 0x98, 0xf1, */
							  /*0xb8, 0xb4, 0x5d, 0xe5, */
							  /*0x1e, 0x41, 0x52, 0x30};*/
// after shiftrow
	/*unsigned char plain[] = { 0xd4, 0xbf, 0x5d, 0x30, */
							  /*0xe0, 0xb4, 0x52, 0xae, */
							  /*0xb8, 0x41, 0x11, 0xf1, */
							  /*0x1e, 0x27, 0x98, 0xe5};*/

	// after mixcolumns
	/*unsigned char plain[] = { 0x04, 0x66, 0x81, 0xe5, */
							  /*0xe0, 0xcb, 0x19, 0x9a, */
							  /*0x48, 0xf8, 0xd3, 0x7a, */
							  /*0x28, 0x06, 0x26, 0x4c };*/
		
	unsigned char plain1[] = { 0x32, 0x43, 0xf6, 0xa8,
							  0x88, 0x5a, 0x30, 0x8d,
							  0x31, 0x31, 0x98, 0xa2,
							  0xe0, 0x37, 0x07, 0x34};
	unsigned char plain2[16];

	ubyte_t key[] = { 0x2b, 0x7e, 0x15, 0x16,
					  0x28, 0xae, 0xd2, 0xa6,
					  0xab, 0xf7, 0x15, 0x88,
					  0x09, 0xcf, 0x4f, 0x3c };

	int i;
	/*ubyte_t key_pool[176];*/

	/*KeyExpansion(key, key_pool, 4, 4, 10);*/
	for (i = 0; i < 16; ++i) {
		printf("0x%02x, ", plain[i]);
		if (!((i+1) % 4))
			printf("\n");
	}
	printf("\n");
	/*AddRoundKey(plain, key_pool+16, 16);*/
	AESEncrypt(plain, key, plain1, 4, 4, 10);
	for (i = 0; i < 16; ++i) {
		printf("0x%02x, ", plain1[i]);
		if (!((i+1) % 4))
			printf("\n");
	}
	printf("\n");
	AESDEncrypt(plain1, key, plain2, 4, 4, 10);
	for (i = 0; i < 16; ++i) {
		printf("0x%02x, ", plain2[i]);
		if (!((i+1) % 4))
			printf("\n");
	}
	printf("\n");
	return 0;

}
