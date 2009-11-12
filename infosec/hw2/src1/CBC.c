#define NK 4
#define NB 4
#define NR 10

#include <string.h>
#include <malloc.h>

#include "Text_Operations.h"
#include "AES.h"
#include "CBC.h"

/*#define CBC_DEBUG*/

int CBCEncrypt(const ubyte_t *plain, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **cipher, int length)
{
	int block_num;
	int i;
	ubyte_t * ptext;

	block_num = FormatPlainText(plain, cipher, length);

	ptext = *cipher;

	//encrypt the first block
	XORText(ptext, v_array, BLOCK_BYTE_SIZE);
	AESEncrypt(ptext, key, NB, NK, NR);

	// the rest blocks
	for (i = 1; i < block_num; ++i) {
		XORText(ptext+(i*BLOCK_BYTE_SIZE), ptext+((i-1)*BLOCK_BYTE_SIZE), BLOCK_BYTE_SIZE);
		AESEncrypt(ptext+(i*BLOCK_BYTE_SIZE), key, NB, NK, NR);
	}

	return block_num * BLOCK_BYTE_SIZE;
}

int CBCDecrypt(const ubyte_t *cipher, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **plain, int length)
{
	int plain_length;
	ubyte_t * ptemp;
	int i;
	int block_num;

	ptemp = (ubyte_t *) malloc(sizeof(ubyte_t)*length);
	memcpy(ptemp, cipher, sizeof(ubyte_t)*length);

	block_num = length / BLOCK_BYTE_SIZE;
	for (i = 0; i < block_num; ++i) {
		AESDecrypt(ptemp+(i*BLOCK_BYTE_SIZE), key, NB, NK, NR);
	}

	XORText(ptemp, v_array, BLOCK_BYTE_SIZE);
	for (i = 1; i < block_num; ++i) {
		XORText(ptemp+i*BLOCK_BYTE_SIZE, cipher+(i-1)*BLOCK_BYTE_SIZE, BLOCK_BYTE_SIZE);
	}

	plain_length = ParsePlainText(ptemp, plain, length);
	free(ptemp);
	return plain_length;
}

