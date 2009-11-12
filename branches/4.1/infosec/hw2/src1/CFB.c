#define NK 4
#define NB 4
#define NR 10

#include <string.h>
#include <malloc.h>

#include "Text_Operations.h"
#include "AES.h"
#include "CFB.h"

/*#define CBC_DEBUG*/

int CFBEncrypt(const ubyte_t *plain, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **cipher, int length)
{
	int i;
	ubyte_t *ptext;
	ubyte_t *v_temp1;
	ubyte_t *v_temp2;

	// alloc the cipher text space
	ptext = (ubyte_t *) malloc(sizeof(ubyte_t) * length);
	/*memcpy(ptext, plain, length);*/
	*cipher = ptext;

	// initial the shift register space
	v_temp1 = (ubyte_t *) malloc(sizeof(ubyte_t) * BLOCK_BYTE_SIZE);
	v_temp2 = (ubyte_t *) malloc(sizeof(ubyte_t) * BLOCK_BYTE_SIZE);
	memcpy(v_temp1, v_array, BLOCK_BYTE_SIZE);

	//encrypt the first 8 bit (1 byte)
	memcpy(v_temp2, v_temp1, BLOCK_BYTE_SIZE);
	AESEncrypt(v_temp2, key, NB, NK, NR);
	ptext[0] = plain[0] ^ v_temp2[0];

	// the rest bytes
	for (i = 1; i < length; ++i) {
		memmove(v_temp1, v_temp1+1, BLOCK_BYTE_SIZE-1);
		v_temp1[BLOCK_BYTE_SIZE-1] = ptext[i-1];
		memcpy(v_temp2, v_temp1, BLOCK_BYTE_SIZE);
		AESEncrypt(v_temp2, key, NB, NK, NR);
		ptext[i] = plain[i] ^ v_temp2[0];
	}

	free(v_temp1);
	free(v_temp2);
	return length;
}

int CFBDecrypt(const ubyte_t *cipher, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **plain, int length)
{
	ubyte_t *ptemp;
	ubyte_t *v_temp1;
	ubyte_t *v_temp2;
	int i;

	ptemp = (ubyte_t *) malloc(sizeof(ubyte_t)*length);
	memcpy(ptemp, cipher, sizeof(ubyte_t)*length);

	// initial the shift register space
	v_temp1 = (ubyte_t *) malloc(sizeof(ubyte_t) * BLOCK_BYTE_SIZE);
	v_temp2 = (ubyte_t *) malloc(sizeof(ubyte_t) * BLOCK_BYTE_SIZE);
	memcpy(v_temp1, v_array, BLOCK_BYTE_SIZE);

	// Decrypt the first 8 bit ( 1 byte)
	memcpy(v_temp2, v_temp1, BLOCK_BYTE_SIZE);
	AESEncrypt(v_temp2, key, NB, NK, NR);
	ptemp[0] = cipher[0] ^ v_temp2[0];


	for (i = 1; i < length; ++i) {
		memmove(v_temp1, v_temp1+1, BLOCK_BYTE_SIZE-1);
		v_temp1[BLOCK_BYTE_SIZE-1] = cipher[i-1];
		memcpy(v_temp2, v_temp1, BLOCK_BYTE_SIZE);
		AESEncrypt(v_temp2, key, NB, NK, NR);
		ptemp[i] = cipher[i] ^ v_temp2[0];
	}

	*plain = ptemp;
	free(v_temp1);
	free(v_temp2);
	return length;
}

