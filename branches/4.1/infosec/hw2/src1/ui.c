#include <stdio.h>
#include <string.h>

#include "key.h"
#include "ECB.h"
#include "CBC.h"
#include "CFB.h"
#include "OFB.h"

#define USAGE "Usage: ./crypt -m mode -e[-d] < in_data.txt > out_data.txt"

#define DATA_BUF 102400

ubyte_t *indata_buf[DATA_BUF];


enum CryptMode { MINVALID = -1, ECB = 0, CBC, CFB, OFB, CTR } crypt_mode;
enum CryptDirection { DINVALID = -1, ENCRYPT = 0, DECRYPT } crypt_direction;

const char *modeName[] = {"ECB", "CBC", "CFB", "OFB", "CTR"};

const char *getModeName(int mode)
{
	return modeName[mode];
}

int main(int argc, char *argv[])
{
	ubyte_t *org_data;
	ubyte_t *c_data;
	ubyte_t *p_data;
	int i;
	int in_data_len;
	ubyte_t c_temp;
	int c_len;
	int p_len;
	int mode;
	int direction;
	int invalid_para;
	char *para_temp;
	FILE *f1, *f2, *f3;

	invalid_para = 0;
	mode = MINVALID;
	direction = DINVALID;
	for (i = 1; i < argc; i++) {
		if(argv[i][0] == '-') {
			if (argv[i][1] == 'm') {
				i++;
				if (i >= argc) {
					fprintf(stderr, "Invalid parameter for '-m'\n");
					invalid_para = 1;
					break;
				}

				para_temp = strdup(argv[i]);
				strupr(para_temp);

				if (strcmp(para_temp, "ecb") == 0) {
					mode = ECB;
				} else if (strcmp(para_temp, "cbc") == 0) {
					mode = CBC;
				} else if (strcmp(para_temp, "cfb") == 0) {
					mode = CFB;
				} else if (strcmp(para_temp, "ofb") == 0) {
					mode = OFB;
				} else if (strcmp(para_temp, "ctr") == 0) {
					mode = CTR;
				} else {
					mode = MINVALID;
					invalid_para = 1;
					fprintf(stderr, "Invalid parameter for '-m'\n");
					free(para_temp);
					break;
				}
				free(para_temp);

			} else if (argv[i][1] == 'd') {
				direction = crypt_direction.DECRYPT;
			} else if (argv[i][1] == 'e') {
				direction = crypt_direction.Encrypt;
			} else {
				fprintf(stderr, "Unrecognized option '%s'\n", argv[i]);
				invalid_para = 1;
				break;
			}
		}
	}

	if (invalid_para ||
			mode == crypt_mode.INVALID ||
			direction == crypt_direction.INVALID) {
		printf("%s\n", USAGE);
		return 10;
	}

	in_data_len = 0;
	while ((c_temp = getchar()) != EOF)
		indata_buf[in_data_len++] = c_temp;

	// ENCRYPT PROCESS
	if (direction == crypt_direction.ENCRYPT) {
		fprintf(stderr, "Encrypting data with %s mode...\n", getModeName(mode));
		switch (mode) {
			case crypt_mode.ECB:
				c_len = ECBEncrypt(indata_buf, test_key, &c_data, in_data_len);
				break;
			case crypt_mode.CBC:
				c_len = CBCEncrypt(indata_buf, test_key, test_V, &c_data, in_data_len);
				break;
			case crypt_mode.CFB:
				c_len = CFBEncrypt(indata_buf, test_key, test_V, &c_data, in_data_len);
				break;
			case crypt_mode.OFB:
				c_len = OFBEncrypt(indata_buf, test_key, &c_data, in_data_len);
				break;
			case crypt_mode.CTR:
				c_len = CTREncrypt(indata_buf, test_key, &c_data, in_data_len);
				break;
		}
		fprintf(stderr, "Encrypt finished. Cipher text length is %d\n");

		for (i = 0; i < c_len; i++)
			fprintf(stdout, "%c", c_data[i]);

		free(c_data);

	} else {
		fprintf(stderr, "Decrypting data with %s mode...\n", getModeName(mode));
		switch (mode) {
			case crypt_mode.ECB:
				p_len = ECBDecrypt(indata_buf, test_key, &p_data, in_data_len);
				break;
			case crypt_mode.CBC:
				p_len = CBCDecrypt(indata_buf, test_key, test_V, &p_data, in_data_len);
				break;
			case crypt_mode.CFB:
				p_len = CFBDecrypt(indata_buf, test_key, test_V, &p_data, in_data_len);
				break;
			case crypt_mode.OFB:
				p_len = OFBDecrypt(indata_buf, test_key, test_V, &p_data, in_data_len);
				break;
			case crypt_mode.CTR:
				p_len = CTRDecrypt(indata_buf, test_key, test_V, &p_data, in_data_len);
				break;
		}
		fprintf(stderr, "Decrypt finished. Plain text length is %d\n");

		for (i = 0; i < p_len; i++)
			fprintf(stdout, "%c", p_data[i]);

		free(p_data);
	}

	return 0;
}
