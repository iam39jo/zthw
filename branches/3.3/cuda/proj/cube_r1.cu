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


__device__ float distance(struct axis *p1, struct axis *p2)
{
	return (p1->x - p2->x) * (p1->x - p2->x) +
		(p1->y - p2->y) * (p1->y - p2->y) +
		(p1->z - p2->z) * (p1->z - p2->z);
}

__global__ void threadCode(int count, float radius2, struct axis *points, float *result)
{
	int thId;
	int i;
	/*float tmp_rst;*/
	int base_idx;
	int tmp_idx;

	__shared__ struct axis block_elem[BLOCK_SIZE];
	__shared__ struct axis sh_data[SHARE_CACHE];
	__shared__ float sh_rst[BLOCK_SIZE];

	thId = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	sh_rst[threadIdx.x] = 0;

	for (base_idx = 0; base_idx <= count; base_idx += SHARE_CACHE) {

		int up_lim = count - base_idx;
		up_lim = up_lim>SHARE_CACHE ? SHARE_CACHE : up_lim;

		__syncthreads();

		/*tmp_idx = threadIdx.x*2;*/
		/*sh_data[tmp_idx] = points[base_idx+tmp_idx];*/
		/*++tmp_idx;*/
		/*sh_data[tmp_idx] = points[base_idx+tmp_idx];*/
		if (threadIdx.x == 0) {
			for (tmp_idx = 0; tmp_idx < SHARE_CACHE; tmp_idx++) {
				sh_data[tmp_idx] = points[base_idx+tmp_idx];
			}
		}
		__syncthreads();

		if (thId >= count)
			return;
		block_elem[threadIdx.x] = points[thId];


		/*tmp_rst = 0;*/
		for (i = 0; i < up_lim; i++) {
			if ((base_idx+i) != thId &&
					distance(&block_elem[threadIdx.x], &sh_data[i]) <= radius2) {
				sh_rst[threadIdx.x] += sh_data[i].v;
			}
		}
	}
	result[thId] = sh_rst[threadIdx.x];
}

int paralize(int count, float radius, struct axis *points, float *sum)
{
	float *cudaRst;
	struct axis *cudaPtr;

	cudaMalloc((void **)&cudaPtr, sizeof(struct axis)*count);
	cudaMalloc((void **)&cudaRst, sizeof(float)*count);

	cudaMemcpy(cudaPtr, points, sizeof(struct axis)*count, cudaMemcpyHostToDevice);
	cudaMemset(cudaRst, 0x0, sizeof(float)*count);

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((count+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

	threadCode<<<dimGrid, dimBlock>>>(count, radius*radius, cudaPtr, cudaRst);
	
	cudaMemcpy(sum, cudaRst, sizeof(float)*count, cudaMemcpyDeviceToHost);
	cudaFree(cudaPtr);
	cudaFree(cudaRst);
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
	paralize(point_count, radius, points, sum);
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
	return 0;
}
