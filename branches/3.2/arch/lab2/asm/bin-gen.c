#include <stdio.h>

#define DELIMITER 1073741824


unsigned int buffer[10000];
int pos;
int data_len;
int code_len;

void gen_data_seg(FILE *fin);

void gen_code_seg(FILE *fin);

int main(int argc, char *argv[])
{
	FILE *fin;
	FILE *fout;

	if (argc < 3) {
		printf("Usage: bin-gen objfile binfile\n\
				\rGenerate executable file for simips machine\n");
		return 2;
	}

	data_len = 0;
	code_len = 0;
	pos = 2;

	fin = fopen(argv[1], "r");
	if (!fin) {
		fprintf(stderr, "Error open file %s\n", argv[1]);
		return 3;
	}
	gen_data_seg(fin);
	gen_code_seg(fin);
	fclose(fin);

	buffer[0] = data_len+code_len;
	buffer[1] = (data_len<<2)+0x200;

	fout = fopen(argv[2], "wb");
	fwrite((void *)buffer, sizeof(unsigned int), pos, fout);
	fclose(fout);

	return 0;
}

void gen_data_seg(FILE *fin)
{
	unsigned int temp;
	do {
		fscanf(fin, "%u", &temp);
		if (temp == DELIMITER)
			return;
		buffer[pos++] = temp;
		data_len++;
	} while (1);
}

void gen_code_seg(FILE *fin)
{
	unsigned int temp;
	do {
		fscanf(fin, "%u", &temp);
		if (temp == DELIMITER)
			return;
		buffer[pos++] = temp;
		code_len++;
	} while (1);
}
