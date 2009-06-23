#include <stdio.h>

#define DELIMITER 1073741824


unsigned int buffer[10000];
int pos;
int data_len;
int code_len;

void gen_data_seg(FILE *fin);

void gen_code_seg(FILE *fin);

int main()
{
	FILE *fin;
	FILE *fout;

	data_len = 0;
	code_len = 0;
	pos = 2;

	fin = fopen("my.txt", "r");
	gen_data_seg(fin);
	gen_code_seg(fin);
	fclose(fin);

	buffer[0] = data_len+code_len;
	buffer[1] = (data_len<<2)+0x200;

	fout = fopen("my.bin", "wb");
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
