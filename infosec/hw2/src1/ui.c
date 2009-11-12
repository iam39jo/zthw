#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>

#include "key.h"
#include "ECB.h"
#include "CBC.h"
#include "CFB.h"
#include "OFB.h"

#define USAGE "Usage: ./aes -m mode -e[-d] -i in_data.txt -o out_data.txt\n\tAvailable mode: ECB/CBC/CFB/OFB/CTR"

#define DATA_BUF 102400

enum CryptMode { MINVALID = -1, ECB = 0, CBC, CFB, OFB, CTR } crypt_mode;
enum CryptDirection { DINVALID = -1, ENCRYPT = 0, DECRYPT } crypt_direction;

const char *modeName[] = {"ECB", "CBC", "CFB", "OFB", "CTR"};

const char *getModeName(int mode)
{
	return modeName[mode];
}

int openInFIle(const char *name, ubyte_t **buf);

int writeToFile(const char *name, const ubyte_t *buf, int size);

int main(int argc, char *argv[])
{
	ubyte_t *org_data;
	ubyte_t *c_data;
	ubyte_t *p_data;
	int i;
	int in_data_len;
	int c_temp;
	int c_len;
	int p_len;
	int mode;
	int direction;
	int in_file;
	const char *in_filename;
	int out_file;
	const char *out_filename;
	int invalid_para;
	char *para_temp;
	FILE *f1, *f2, *f3;
	double time_start;
	double time_cost;

	invalid_para = 0;
	mode = MINVALID;
	direction = DINVALID;
	in_file = 0;
	out_file = 0;
	for (i = 1; i < argc; i++) {
		if(argv[i][0] == '-') {
			if (argv[i][1] == 'm' || argv[i][1] == 'M') {
				i++;
				if (i >= argc) {
					fprintf(stderr, "Invalid parameter for '-m'\n");
					invalid_para = 1;
					break;
				}

				para_temp = strdup(argv[i]);
				in_data_len = strlen(para_temp);
				while (--in_data_len >= 0)
					para_temp[in_data_len] = tolower(para_temp[in_data_len]);

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

			} else if (argv[i][1] == 'd' || argv[i][1] == 'D') {
				direction = DECRYPT;

			} else if (argv[i][1] == 'e' || argv[i][1] == 'E') {
				direction = ENCRYPT;

			} else if (argv[i][1] == 'i' || argv[i][1] == 'I') {
				i++;
				if (i >= argc) {
					fprintf(stderr, "Invalid parameter for '-i'\n");
					invalid_para = 1;
					break;
				}

				in_file = 1;
				in_filename = argv[i];

			} else if (argv[i][1] == 'o' || argv[i][1] == 'O') {
				i++;
				if (i >= argc) {
					fprintf(stderr, "Invalid parameter for '-o'\n");
					invalid_para = 1;
					break;
				}

				out_file = 1;
				out_filename = argv[i];

			} else {
				fprintf(stderr, "Unrecognized option '%s'\n", argv[i]);
				invalid_para = 1;
				break;
			}
		}
	}

	if (invalid_para ||	mode == MINVALID || direction == DINVALID ||
			in_file == 0 || out_file == 0) {
		printf("%s\n", USAGE);
		return 10;
	}

	in_data_len = openInFIle(in_filename, &org_data);

	if (in_data_len == -1) {
		printf("%s\n", USAGE);
		return 11;
	}

	time_start = clock();

	// ENCRYPT PROCESS
	if (direction == ENCRYPT) {
		fprintf(stderr, "Encrypting data with %s mode...\n", getModeName(mode));
		switch (mode) {
			case ECB:
				c_len = ECBEncrypt(org_data, test_key, &c_data, in_data_len);
				break;
			case CBC:
				c_len = CBCEncrypt(org_data, test_key, test_V, &c_data, in_data_len);
				break;
			case CFB:
				c_len = CFBEncrypt(org_data, test_key, test_V, &c_data, in_data_len);
				break;
			case OFB:
				c_len = OFBEncrypt(org_data, test_key, test_V, &c_data, in_data_len);
				break;
			case CTR:
				c_len = CTREncrypt(org_data, test_key, test_V, &c_data, in_data_len);
				break;
		}
		fprintf(stderr, "Encrypt finished. Cipher text length is %d\n", c_len);

		/*for (i = 0; i < c_len; i++)*/
			/*fprintf(stdout, "%c", c_data[i]);*/
		writeToFile(out_filename, c_data, c_len);
		free(c_data);
	} else {
		fprintf(stderr, "Decrypting data with %s mode...\n", getModeName(mode));
		switch (mode) {
			case ECB:
				p_len = ECBDecrypt(org_data, test_key, &p_data, in_data_len);
				break;
			case CBC:
				p_len = CBCDecrypt(org_data, test_key, test_V, &p_data, in_data_len);
				break;
			case CFB:
				p_len = CFBDecrypt(org_data, test_key, test_V, &p_data, in_data_len);
				break;
			case OFB:
				p_len = OFBDecrypt(org_data, test_key, test_V, &p_data, in_data_len);
				break;
			case CTR:
				p_len = CTRDecrypt(org_data, test_key, test_V, &p_data, in_data_len);
				break;
		}
		fprintf(stderr, "Decrypt finished. Plain text length is %d\n", p_len);

		/*for (i = 0; i < p_len; i++)*/
			/*fprintf(stdout, "%c", p_data[i]);*/
		writeToFile(out_filename, p_data, p_len);
		free(p_data);
	}

	time_cost = (clock() - time_start) / CLOCKS_PER_SEC;

	fprintf(stderr, "Time cost: %lf (s)\n", time_cost);

	free(org_data);
	return 0;
}

int openInFIle(const char *name, ubyte_t **buf)
{
	FILE *fp;
	int fsize;
	ubyte_t *ptemp;

	fp = fopen(name, "rb");
	if (fp == NULL) {
		fprintf(stderr, "Cannot open input file '%s'\n", name);
		return -1;
	}

	fseek(fp, 0, SEEK_END);
	fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	ptemp = (ubyte_t *) malloc(sizeof(ubyte_t) * fsize);

	fread(ptemp, sizeof(ubyte_t), fsize, fp);
	fclose(fp);
	*buf = ptemp;
	return fsize;
}

int writeToFile(const char *name, const ubyte_t *buf, int size)
{
	FILE *fp;

	fp = fopen(name, "wb");
	if (fp == NULL) {
		fprintf(stderr, "Cannot open output file '%s'\n", name);
		return -1;
	}

	fwrite(buf, sizeof(ubyte_t), size, fp);
	fclose(fp);
	return;
}
