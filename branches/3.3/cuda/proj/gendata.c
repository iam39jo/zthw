#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define POINT_COUNT 10000000

int main(int argc, char * argv[])
{
	float x, y, z,value;
	float r = 0.02;
	int i;
	FILE *fp;

	if (argc >= 2) {
		fp = fopen(argv[1], "w");
	} else {
		printf("Usage: gendata filename\n");
		exit(1);
	}

	fprintf(fp, "%d %f\n", POINT_COUNT, r);

	srand(time(NULL));

	for (i = 0; i < POINT_COUNT; i++) {
		x = 1.0*rand()/RAND_MAX;
		y = 1.0*rand()/RAND_MAX;
		z = 1.0*rand()/RAND_MAX;
		value = 1.0*rand()/RAND_MAX*100;
		fprintf(fp, "%f %f %f %f\n", x, y, z, value);
	}

	fclose(fp);
	return 0;
}


