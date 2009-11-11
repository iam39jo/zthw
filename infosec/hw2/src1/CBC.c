#define NK 4
#define NB 4
#define NR 10

#include <string.h>
#include <malloc.h>

#include "Text_Operations.h"
#include "AES.h"

/*#define CBC_DEBUG*/

int CBCEncrypt(const ubyte_t *plain, const ubyte_t *key, const ubyte_t *v_array, ubyte_t **cipher, int length)
{
	int block_num;
	int i;
	ubyte_t * ptext;

	block_num = FormatPlainText(plain, cipher, length);

	ptext = *cipher;

	XORText(ptext, v_array, BLOCK_BYTE_SIZE);

	//encrypt the first block
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
	
	plain_length = ParsePlainText(ptemp, plain, length);
	free(ptemp);
	return plain_length;
}

#ifdef ECB_DEBUG

#include "key.h"
int main()
{
	ubyte_t data[1024];
	ubyte_t *c_data;
	ubyte_t *p_data;
	int c_len;
	int p_len;
	int i;

	printf("ORG:\n");
	for (i = 0; i < 1024; i++) {
		printf(" %02X", data[i]);
		if (!(i % 25))
			printf("\n");
	}
	printf("\n");

	c_len = ECBEncrypt(data, test_key, &c_data, 1024);
	printf("C_DATA:\n");
	for (i = 0; i < c_len; i++) {
		printf(" %02X", c_data[i]);
		if (!(i % 25))
			printf("\n");
	}
	printf("\n");

	p_len = ECBDecrypt(c_data, test_key, &p_data, 1024);
	printf("P_DATA:\n");
	for (i = 0; i < p_len; i++) {
		printf(" %02X", data[i]);
		if (!(i % 25))
			printf("\n");
	}
	printf("\n");
	return 0;
}

#endif
