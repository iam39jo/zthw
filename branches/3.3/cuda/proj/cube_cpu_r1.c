#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/*
 * WORLD_SIZE	The length of the whole world's edge
 * R			The length of the seperated cubes' edge
 * CUBE_NUM		Total number of seperated cubes(should equal to (WORLD_SIZE^3 / R^3)
 * CUBE_PER_EDGE	Cubes on one edge (WORLD_SIZE / R)
 */
#define WORLD_SIZE 1
#define R 0.1
#define CUBE_NUM 1000
#define CUBE_PER_EDGE 10

struct axis {
	float x;
	float y;
	float z;
	float v;
};

struct cube_info {
	int start;
	int length;
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

int cmp_x(const void *a, const void *b)
{
	return (*(struct axis **)a)->x < (*(struct axis **)b)->x ? -1 : 1;
}

int cmp_y(const void *a, const void *b)
{
	return (*(struct axis **)a)->y < (*(struct axis **)b)->y ? -1 : 1;
}

int cmp_z(const void *a, const void *b)
{
	return (*(struct axis **)a)->z < (*(struct axis **)b)->z ? -1 : 1;
}

int violent(int count, float radius, struct axis *points, float *sum)
{
	int i, j;

	float radius2 = radius*radius;
	float up_lim;
	int cube_idx;

	struct axis **ptr_array = (struct axis **) malloc(sizeof(struct axis *) * count);

	for (i = 0; i < count; i++)
		ptr_array[i] = &points[i];

	qsort(ptr_array, count, sizeof(struct axis *), cmp_x);
	for (i = 0, up_lim = R, cube_idx = 0; i < count; i++) {
		if (ptr_array[i].x <= up_lim)




	qsort(ptr_array, count, sizeof(struct axis *), cmp_y);
	qsort(ptr_array, count, sizeof(struct axis *), cmp_z);

	for (i = 0; i < count; i++)
		printf("%f %f %f %f\n", ptr_array[i]->x, ptr_array[i]->y, ptr_array[i]->z, ptr_array[i]->v);

	return 0;

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
