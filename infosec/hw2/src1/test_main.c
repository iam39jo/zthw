#include <stdio.h>

#include "key.h"
#include "ECB.h"

#define TEST_DATA_LEN 1024

int main()
{
	ubyte_t data[TEST_DATA_LEN];
	ubyte_t *c_data;
	ubyte_t *p_data;
	int i;
	FILE *f1, *f2, *f3;

	f1 = fopen("ORG.dat", "wb");
	f2 = fopen("C_DATA.dat", "wb");
	f3 = fopen("P_DATA.dat", "wb");

	/*printf("ORG:\n");*/
	for (i = 0; i < TEST_DATA_LEN; i++) {
		fprintf(f1, " %02X", data[i]);
		if (i % 30 == 29)
			fprintf(f1, "\n");
	}
	printf("\n");

	ECBEncrypt(data, test_key, &c_data, TEST_DATA_LEN);
	printf("Encrypt done\n");
	for (i = 0; i < TEST_DATA_LEN; i++) {
		fprintf(f2, " %02X", c_data[i]);
		if (i % 30 == 29)
			fprintf(f2, "\n");
	}
	printf("\n");

	ECBDecrypt(c_data, test_key, &p_data, TEST_DATA_LEN);
	printf("Decrypt done\n");
	for (i = 0; i < TEST_DATA_LEN; i++) {
		fprintf(f3, " %02X", data[i]);
		if (i % 30 == 29)
			fprintf(f3, "\n");
	}
	printf("\n");

	fclose(f1);
	fclose(f2);
	fclose(f3);
	return 0;
}
