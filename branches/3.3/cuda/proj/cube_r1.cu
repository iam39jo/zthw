#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define DB(msg, args...) fprintf(stderr, "[DEBUG](%s:%d-%s):" msg "\n", __FILE__, __LINE__, __FUNCTION__, ##args)

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

#define BLOCK_SIZE 256

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

struct axis *orig_data;

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


__device__ float distance2(struct axis *p1, struct axis *p2)
{
	return (p1->x - p2->x) * (p1->x - p2->x) +
		(p1->y - p2->y) * (p1->y - p2->y) +
		(p1->z - p2->z) * (p1->z - p2->z);
}

__global__ void threadCode(int count, int range, float radius2, struct axis *points, float *result, struct cube_info ***cubes)
{
	__shared__ int lower_lim, upper_lim;
	__shared__ int x_lower, x_upper, y_lower, y_upper, z_lower, z_upper;
	int cube_x, cube_y, cube_z;
	int i;

	lower_lim = cubes[blockIdx.x][blockIdx.y][blockIdx.z].start;
	upper_lim = cubes[blockIdx.x][blockIdx.y][blockIdx.z].start+cubes[blockIdx.x][blockIdx.y][blockIdx.z].length;
	x_lower = blockIdx.x - range; x_lower = x_lower < 0 ? 0 : x_lower;
	y_lower = blockIdx.y - range; y_lower = y_lower < 0 ? 0 : y_lower;
	z_lower = blockIdx.z - range; z_lower = z_lower < 0 ? 0 : z_lower;
	x_upper = blockIdx.x + range; x_upper = x_upper < CUBE_PER_EDGE ? x_upper : CUBE_PER_EDGE-1;
	y_upper = blockIdx.y + range; y_upper = y_upper < CUBE_PER_EDGE ? y_upper : CUBE_PER_EDGE-1;
	z_upper = blockIdx.z + range; z_upper = z_upper < CUBE_PER_EDGE ? z_upper : CUBE_PER_EDGE-1;

	__syncthreads();

	for (i = lower_lim; i < upper_lim; i++)

	/*thId = threadIdx.x + blockIdx.x*BLOCK_SIZE;*/
	/*sh_rst[threadIdx.x] = 0;*/
}

int cmp_x(const void *a, const void *b)
{
	return orig_data[(*(int *)a)].x < orig_data[(*(int *)b)].x ? -1 : 1;
}

int cmp_y(const void *a, const void *b)
{
	return orig_data[(*(int *)a)].y < orig_data[(*(int *)b)].y ? -1 : 1;
	/*return (*(struct axis **)a)->y < (*(struct axis **)b)->y ? -1 : 1;*/
}

int cmp_z(const void *a, const void *b)
{
	return orig_data[(*(int *)a)].z < orig_data[(*(int *)b)].z ? -1 : 1;
	/*return (*(struct axis **)a)->z < (*(struct axis **)b)->z ? -1 : 1;*/
}

void divideIntoCubes(int count, int *idx_array, struct axis *points, struct cube_info cubes[CUBE_PER_EDGE][CUBE_PER_EDGE][CUBE_PER_EDGE])
{
	float up_lim;
	int cube_uplim;
	int i, j, k;
	int cube_idx;
	int len;

	qsort(idx_array, count, sizeof(int), cmp_x);
	/*for (i = 0; i < count; i++)*/
		/*DB("%f %f %f", points[idx_array[i]].x, points[idx_array[i]].y, points[idx_array[i]].z);*/
	up_lim = R;
	cube_idx = 0;
	len = 0;
	cubes[0][0][0].start = 0;
	for (i = 0; i < count; i++) {
		if (points[idx_array[i]].x <= up_lim) {
			len++;
		} else {
			if (up_lim >= 1)						//debug
				exit(1);
			up_lim += R;
			//the length of previous cube block
			cubes[cube_idx][0][0].length = len;
			cube_idx++;
			// the start idx of next cube block
			if (cube_idx >= CUBE_PER_EDGE)
				exit(2);
			cubes[cube_idx][0][0].start = i;
			len = 0;
			i--;
		}
	}
	// update the latest length
	cubes[cube_idx][0][0].length = len;
	cube_idx++;
	// and the rest empty block
	for ( ; cube_idx < CUBE_PER_EDGE; cube_idx++) {
		if (cube_idx == 0) {
			cubes[cube_idx][0][0].length = 0;
		} else {
			cubes[cube_idx][0][0].start = i-1;
			cubes[cube_idx][0][0].length = 0;
		}
	}
			

	for (i = 0; i < CUBE_PER_EDGE; i++) {
		qsort(&idx_array[cubes[i][0][0].start], cubes[i][0][0].length, sizeof(int), cmp_y);
		up_lim = R;
		cube_idx = 0;
		len = 0;
		cube_uplim = cubes[i][0][0].start + cubes[i][0][0].length;
		for (j = cubes[i][0][0].start; j < cube_uplim; j++) {
			if (points[idx_array[j]].y <= up_lim) {
				len++;
			} else {
				if (up_lim >= 1)				//debug
					exit(1);
				up_lim += R;
				cubes[i][cube_idx][0].length = len;
				cube_idx++;
				if (cube_idx >= CUBE_PER_EDGE)
					exit(2);
				cubes[i][cube_idx][0].start = j;
				len = 0;
				j--;
			}
		}
		cubes[i][cube_idx][0].length = len;
		cube_idx++;
		for ( ; cube_idx < CUBE_PER_EDGE; cube_idx++) {
			if (cube_idx == 0) {
				cubes[i][cube_idx][0].length = 0;
			} else {
				cubes[i][cube_idx][0].start = j-1;
				cubes[i][cube_idx][0].length = 0;
			}
		}
	}
	/*for (i = 0; i < count; i++)*/
		/*DB("%f %f %f", points[idx_array[i]].x, points[idx_array[i]].y, points[idx_array[i]].z);*/

	for (i = 0; i < CUBE_PER_EDGE; i++) {
		for (j = 0; j < CUBE_PER_EDGE; j++) {
			/*DB("%d %d %d %d", i, j, cubes[i][j][0].start, cubes[i][j][0].length);*/
			qsort(&idx_array[cubes[i][j][0].start], cubes[i][j][0].length, sizeof(int), cmp_z);
			up_lim = R;
			cube_idx = 0;
			len = 0;
			cube_uplim = cubes[i][j][0].start + cubes[i][j][0].length;
			for (k = cubes[i][j][0].start; k < cube_uplim; k++) {
				if (points[idx_array[k]].z <= up_lim) {
					len++;
				} else {
					if (up_lim >= 1)
						exit(1);
					up_lim += R;
					cubes[i][j][cube_idx].length = len;
					cube_idx++;
					if (cube_idx >= CUBE_PER_EDGE)
						exit(2);
					cubes[i][j][cube_idx].start = k;
					len = 0;
					k--;
				}
			}
			cubes[i][j][cube_idx].length = len;
			cube_idx++;
			for ( ; cube_idx < CUBE_PER_EDGE; cube_idx++) {
				if (cube_idx == 0) {
					cubes[i][j][cube_idx].length = 0;
				} else {
					cubes[i][j][cube_idx].start = k-1;
					cubes[i][j][cube_idx].length = 0;
				}
			}
		}
	}
	/*for (i = 0; i < count; i++)*/
		/*DB("%f %f %f", points[idx_array[i]].x, points[idx_array[i]].y, points[idx_array[i]].z);*/
}

void copyCubes(struct cube_info ***cudaCubes, struct cube_info cubes[CUBE_PER_EDGE][CUBE_PER_EDGE][CUBE_PER_EDGE])
{
	int i, j;
	void *tmpPtr1[CUBE_PER_EDGE];
	void *tmpPtr2[CUBE_PER_EDGE];

	for (i = 0; i < CUBE_PER_EDGE; i++) {
		cudaMalloc(&tmpPtr1[i], sizeof(struct cube_info *) * CUBE_PER_EDGE);
		
		for (j = 0; j < CUBE_PER_EDGE; j++) {
			cudaMalloc(&tmpPtr2[j], sizeof(struct cube_info) * CUBE_PER_EDGE);
			cudaMemcpy(tmpPtr2[j], (void *) cubes[i][j], sizeof(struct cube_info) * CUBE_PER_EDGE, cudaMemcpyHostToDevice);
		}

		cudaMemcpy(tmpPtr1[i], (void *) tmpPtr2, sizeof(struct cube_info *) * CUBE_PER_EDGE, cudaMemcpyHostToDevice);
	}
		cudaMemcpy((void *) cudaCubes, (void *) tmpPtr1, sizeof(struct cube_info **) * CUBE_PER_EDGE, cudaMemcpyHostToDevice);
}

void freeCubes(struct cube_info ***cudaCubes)
{
	return;
}

int paralize(int count, float radius, struct axis *points, float *results)
{
	float *cudaRst;
	struct axis *cudaPtr;
	struct cube_info ***cudaCubes;
	struct axis *tmpPoints;
	float *tmpResult;
	struct cube_info cubes[CUBE_PER_EDGE][CUBE_PER_EDGE][CUBE_PER_EDGE];
	/*struct axis **ptr_array = (struct axis **) malloc(sizeof(struct axis *) * count);*/
	int i;
	int *idx_array = (int *) malloc(sizeof(int) * count);
	tmpPoints = (struct axis *) malloc(sizeof(struct axis) * count);
	tmpResult = (float *) malloc(sizeof(float) * count);

	/*
	   divide points into cubes
   */
	for (i = 0; i < count; i++)
		idx_array[i] = i;
	divideIntoCubes(count, idx_array, points, cubes);
	for (i = 0; i < count; i++)
		tmpPoints[i] = points[idx_array[i]];
	/*for (i = cubes[0][0][0].start; i < cubes[0][0][0].start+cubes[0][0][0].length; i++)*/
		/*DB("%d %f %f %f", i, tmpPoints[i].x, tmpPoints[i].y, tmpPoints[i].z);*/
		

	/*int j, k, z;*/
	/*for (j = 0; j < 10; j++)*/
		/*for (k = 0; k < 10; k++)*/
			/*for (z = 0; z < 10; z++) {*/
				/*DB("%d %d %d(%d %d):", j, k, z, cubes[j][k][z].start, cubes[j][k][z].length);*/
				/*for (i = cubes[j][k][z].start; i < cubes[j][k][z].start+cubes[j][k][z].length; i++)*/
					/*DB("\t%f %f %f %f",points[i].x, points[i].y, points[i].z, points[i].v);*/
			/*}*/

	cudaMalloc((void **)&cudaPtr, sizeof(struct axis)*count);
	cudaMalloc((void **)&cudaRst, sizeof(float)*count);
	cudaMalloc((void **)&cudaCubes, sizeof(struct cube_info **) * CUBE_PER_EDGE);

	cudaMemcpy(cudaPtr, points, sizeof(struct axis)*count, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaCubes, cubes, sizeof(struct cube_info)*count, cudaMemcpyHostToDevice);
	cudaMemset(cudaRst, 0x0, sizeof(float)*count);
	copyCubes(cudaCubes, cubes);

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(CUBE_PER_EDGE, CUBE_PER_EDGE, CUBE_PER_EDGE);
	threadCode<<<dimGrid, dimBlock>>>(count, (radius + R)/R, radius*radius, cudaPtr, cudaRst, cudaCubes);
	
	cudaMemcpy(tmpResult, cudaRst, sizeof(float)*count, cudaMemcpyDeviceToHost);
	for (i = 0; i < count; i++)
		results[idx_array[i]] = tmpResult[i];
	/*cudaMemcpy(points, cudaPtr, sizeof(struct axis)*count, cudaMemcpyDeviceToHost);*/
	freeCubes(cudaCubes);
	cudaFree(cudaPtr);
	cudaFree(cudaRst);
	free(idx_array);
	free(tmpPoints);
	free(tmpResult);
	return 0;
}


int cal(FILE *fp)
{
	int point_count;
	float radius;
	int i;
	struct timeval tv_start, tv_end;
	double time_cost;
	float *results;

	fscanf(fp, "%d %f", &point_count, &radius);

	orig_data = (struct axis *) malloc(sizeof(struct axis)*point_count);
	results = (float *) malloc(sizeof(float)*point_count);
	/*memset((void *) results, 0x0, sizeof(float)*point_count);*/

	for (i = 0; i < point_count; i++) {
		fscanf(fp, "%f %f %f %f", &orig_data[i].x,	&orig_data[i].y, 
				&orig_data[i].z, &orig_data[i].v);
	}

	/* execute calculation and get time stamp */
	gettimeofday(&tv_start, NULL);
	paralize(point_count, radius, orig_data, results);
	gettimeofday(&tv_end, NULL);

	time_cost = 1000000 * (tv_end.tv_sec - tv_start.tv_sec) +
		(tv_end.tv_usec - tv_start.tv_usec);
	time_cost /= 1000000;

	/* output the result and time cost */
	for (i = 0; i < point_count; i++)
		printf("Point %5d: %f\n", i+1, results[i]);
	printf("Time: %lf\n", time_cost);

	free(orig_data);
	free(results);
	return 0;
}
