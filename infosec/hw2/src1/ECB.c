#define NK 4
#define NB 4
#define NR 10

#include <string.h>
#include <malloc.h>

#include "PlainText.h"
#include "AES.h"

int ECBEncrypt(const ubyte_t *plain, const ubyte_t *key, ubyte_t **cipher, int length)
{
	int block_num;
	int i;
	ubyte_t * ptext;

	block_num = FormatPlainText(plain, cipher, length);

	ptext = *cipher;

	for (i = 0; i < block_num; ++i) {
		AESEncrypt(ptext+(i*BLOCK_BYTE_SIZE), key, NB, NK, NR);
	}

	return block_num * BLOCK_BYTE_SIZE;
}

int ECBDecrypt(const ubyte_t *cipher, const ubyte_t *key, ubyte_t **plain, int length)
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
	
	plain_length = ParsePlainText(ptemp, plain, length);
	free(ptemp);
	return plain_length;
}
