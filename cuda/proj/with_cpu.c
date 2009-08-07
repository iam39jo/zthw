#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

struct axis {
	float x;
	float y;
	float z;
	float v;
};

int cal(FILE *fp);

int main(int argc, char *argv[])
{
	FILE *fp;

	if (argc >= 2) {
		fp = fopen(argv[1], "r");

	} else {
		printf("Usage: prog datafile\n");
		exit(1);
	}

	cal(fp);
	fclose(fp);

	return 0;
}

float distance2(struct axis *p1, struct axis *p2)
{
	return (p1->x - p2->x) * (p1->x - p2->x) +
		(p1->y - p2->y) * (p1->y - p2->y) +
		(p1->z - p2->z) * (p1->z - p2->z);
}

int violent(int count, float radius, struct axis *points, float *sum)
{
	int i, j;

	float radius2 = radius*radius;

	for (i = 0; i < count-1; i++) {
		for (j = i+1; j < count; j++) {
			if (distance2(&points[i], &points[j]) <= radius2) {
				sum[i] += points[j].v;
				sum[j] += points[i].v;
			}
		}
	}

}

int cal(FILE *fp)
{
	int point_count;
	float radius;
	struct axis *points;
	int i;
	struct timeval tv_start, tv_end;
	double time_cost;
	float *sum;

	fscanf(fp, "%d %f", &point_count, &radius);

	points = (struct axis *) malloc(sizeof(struct axis)*point_count);
	sum = (float *) malloc(sizeof(float)*point_count);
	memset((void *) sum, 0x0, sizeof(float)*point_count);

	for (i = 0; i < point_count; i++)
		fscanf(fp, "%f %f %f %f", &points[i].x,	&points[i].y, 
				&points[i].z, &points[i].v);

	/* execute calculation and get time stamp */
	gettimeofday(&tv_start, NULL);
	violent(point_count, radius, points, sum);
	gettimeofday(&tv_end, NULL);

	time_cost = 1000000 * (tv_end.tv_sec - tv_start.tv_sec) +
		(tv_end.tv_usec - tv_start.tv_usec);
	time_cost /= 1000000;

	/* output the result and time cost */
	for (i = 0; i < point_count; i++)
		printf("Point %5d: %f\n", i+1, sum[i]);
	printf("Time: %lf\n", time_cost);

	free(points);
	free(sum);
}
