#define NK 4
#define NB 4
#define NR 10

#include <string.h>
#include <malloc.h>

#include "Text_Operations.h"
#include "AES.h"
#include "CTR.h"

void addCounter(ubyte_t *counter, int d, int length);

int CTREncrypt(const ubyte_t *plain, const ubyte_t *key, const ubyte_t *counter, ubyte_t **cipher, int length)
{
	int i;
	int fblock_num;
	int last_size;
	ubyte_t *ptext;
	ubyte_t *ctr_temp1;
	ubyte_t *ctr_temp2;

	// alloc the cipher text space
	ptext = (ubyte_t *) malloc(sizeof(ubyte_t) * length);
	memcpy(ptext, plain, length);
	*cipher = ptext;

	// initial the counter space
	ctr_temp1 = (ubyte_t *) malloc(sizeof(ubyte_t) * BLOCK_BYTE_SIZE);
	ctr_temp2 = (ubyte_t *) malloc(sizeof(ubyte_t) * BLOCK_BYTE_SIZE);
	memcpy(ctr_temp1, counter, BLOCK_BYTE_SIZE);

	fblock_num = length / BLOCK_BYTE_SIZE;
	last_size = length % BLOCK_BYTE_SIZE;

	for (i = 0; i < fblock_num; i++) {
		memcpy(ctr_temp2, ctr_temp1, BLOCK_BYTE_SIZE);
		AESEncrypt(ctr_temp2, key, NB, NK, NR);
		XORText(ptext + i * BLOCK_BYTE_SIZE, ctr_temp2, BLOCK_BYTE_SIZE);
		addCounter(ctr_temp1, 4, BLOCK_BYTE_SIZE);
	}

	// encrypt the last block
	if (last_size > 0) {
		AESEncrypt(ctr_temp1, key, NB, NK, NR);
		XORText(ptext + fblock_num*BLOCK_BYTE_SIZE, ctr_temp1, last_size);
	}

	free(ctr_temp1);
	free(ctr_temp2);
	return length;
}

int CTRDecrypt(const ubyte_t *cipher, const ubyte_t *key, const ubyte_t *counter, ubyte_t **plain, int length)
{
	ubyte_t *ptemp;
	ubyte_t *ctr_temp1;
	ubyte_t *ctr_temp2;
	int fblock_num;
	int last_size;
	int i;

	ptemp = (ubyte_t *) malloc(sizeof(ubyte_t)*length);
	memcpy(ptemp, cipher, sizeof(ubyte_t)*length);

	// initial the shift register space
	ctr_temp1 = (ubyte_t *) malloc(sizeof(ubyte_t) * BLOCK_BYTE_SIZE);
	ctr_temp2 = (ubyte_t *) malloc(sizeof(ubyte_t) * BLOCK_BYTE_SIZE);
	memcpy(ctr_temp1, counter, BLOCK_BYTE_SIZE);

	fblock_num = length / BLOCK_BYTE_SIZE;
	last_size = length % BLOCK_BYTE_SIZE;

	for (i = 0; i < fblock_num; ++i) {
		memcpy(ctr_temp2, ctr_temp1, BLOCK_BYTE_SIZE);
		AESEncrypt(ctr_temp2, key, NB, NK, NR);
		XORText(ptemp + i * BLOCK_BYTE_SIZE, ctr_temp2, BLOCK_BYTE_SIZE);
		addCounter(ctr_temp1, 4, BLOCK_BYTE_SIZE);
	}

	if (last_size > 0) {
		AESEncrypt(ctr_temp1, key, NB, NK, NR);
		XORText(ptemp + fblock_num * BLOCK_BYTE_SIZE, ctr_temp1, BLOCK_BYTE_SIZE);
	}

	*plain = ptemp;
	free(ctr_temp1);
	free(ctr_temp2);
	return length;
}

void addCounter(ubyte_t *counter, int d, int length)
{
	int i;

	for (i = 0; i < length; i++)
		counter[i] += d;
}
