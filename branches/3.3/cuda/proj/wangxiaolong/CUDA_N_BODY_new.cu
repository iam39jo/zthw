#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
int T=-1,RR=-1;
/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
bool InitCUDA(void)
{
	int count = 0;
	int i = 0;
	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}
/************************************************************************/
float r;
const int MAX=10000000;
int threadnum,blocknum,gridnum,sidenum,N,cubenum;
float R,OUT[MAX];
float4 P[MAX];
__global__ void MEMSET(int size,int* arr,int value,int threadnum,int blocknum) {
	int i,j,k,tid,bid,id,step;
	tid=threadIdx.x;
	bid=blockIdx.x;
	id=bid*threadnum+tid;
	step=ceil(float(size)/threadnum/blocknum);
	for(i=step*id;i<size && i<step*(id+1);i++) arr[i]=value;
	__syncthreads();
}
__device__ void device_MEMSET(int size,int* arr,int value,int tid,int threadnum,int blocknum) {
	int i,j,k,step;
	step=ceil(float(size)/threadnum);
	for(i=step*tid;i<size && i<step*(tid+1);i++) arr[i]=value;
}
//calculate the cube id of each partical
//N=gridnum*blocknum*threadnum
//run 1 times, with blocknum blocks, threadnum threads
__global__ void SORT_1(int N,int* count,int* begin,int* belong,int* retrive,float4* oldP,float4* newP,
		  int cubenum,int sidenum,float r,int threadnum,int blocknum) {
	int i,j,k,tid,bid,id,IDx,IDy,IDz,ID,beg,end,step;
	tid=threadIdx.x;
	bid=blockIdx.x;
	id=tid+bid*threadnum;
	step=ceil(float(N)/blocknum/threadnum);
	beg=step*id;
	end=beg+step;
	for(i=beg;i<N && i<end;i++) {
		IDx=oldP[i].x/r;	IDy=oldP[i].y/r;	IDz=oldP[i].z/r;
		ID=IDx+IDy*sidenum+IDz*sidenum*sidenum;
		belong[i]=ID;	begin[ID]++;	count[ID]++;
	}
}
//accmulate vecter begin
//only need to run one time
//need to use chip memory, only thread 0 effective
__global__ void SORT_2(int N,int* count,int* begin,int* belong,int* retrive,float4* oldP,float4* newP,
		  int cubenum,int sidenum,float r,int threadnum,int blocknum) {
	int i,j,k,tid,bid,id;
	if(threadIdx.x==0 && blockIdx.x==0) 
		for(i=1;i<cubenum;i++) begin[i]+=begin[i-1];
}
//place partical to new vector newP, and set retrival vector for future output
//run 1 times, with blocknum blocks, threadnum threads
__global__ void SORT_3(int N,int* count,int* begin,int* belong,int* retrive,float4* oldP,float4* newP,
		  int cubenum,int sidenum,float r,int threadnum,int blocknum) {
	int i,j,k,tid,bid,id,beg,end,step,ID;
	tid=threadIdx.x;
	bid=blockIdx.x;
	id=tid+bid*threadnum;
	step=ceil(float(N)/blocknum/threadnum);
	beg=step*id;
	end=beg+step;
	for(i=beg;i<N && i<end;i++) { ID=belong[i]; j=--begin[ID]; retrive[j]=i; newP[j]=oldP[i]; }
}
//SORT_2 collecting datas. only need to run on 1 thread
void SORT(int N,int* count,int* begin,int* belong,int* retrive,float4* oldP,float4* newP,
		  int cubenum,int sidenum,float r,int threadnum,int blocknum) {
	MEMSET<<<blocknum,threadnum>>>(cubenum,begin,0,threadnum,blocknum);
	MEMSET<<<blocknum,threadnum>>>(cubenum,count,0,threadnum,blocknum);
	SORT_1<<<blocknum,threadnum>>>(N,count,begin,belong,retrive,oldP,newP,cubenum,sidenum,r,threadnum,blocknum);
	SORT_2<<<1,1>>>(N,count,begin,belong,retrive,oldP,newP,cubenum,sidenum,r,threadnum,blocknum);
	SORT_3<<<blocknum,threadnum>>>(N,count,begin,belong,retrive,oldP,newP,cubenum,sidenum,r,threadnum,blocknum);
}
//every time when each kernel runs, it includes blocknum blocks, with the threadnum which is the largest number
//in each practical_cube which each represents a block
//
__global__ void CALC(int sidenum,int threadnum,int blocknum,int* begin,int* count,float* ans,float4* newP,
					 int partical_cube_gid,int partical_cubenum,int* partical_begin,int* partical_count,
					 float R,float r,float4* partical_newP,int partical_sidenum,float partical_r) {
	float eps=0.000002;
	int i,j,k,tid,bid,beg,end,step,access,_x,_y,_z,partical_cube_id;
	float sum=0;
	bool valid=true;
	float4 _POINT;
	__shared__ float max_x,max_y,max_z,min_x,min_y,min_z;
	__shared__ int xid,Xid,yid,Yid,zid,Zid,readnum,totalnum;
	tid=threadIdx.x;
	bid=blockIdx.x;
	partical_cube_id=partical_cube_gid*blocknum+bid;
	//empty block, return directly
	if(partical_cube_id>=partical_cubenum || partical_count[partical_cube_id]==0) return;
	if(tid<partical_count[partical_cube_id]) 
		_POINT=partical_newP[partical_begin[partical_cube_id]+tid]; 
	else valid=false;
	if(tid==0) max_x=max_y=max_z=-1, min_x=min_y=min_z=1;
	__syncthreads();
	//every threads calculate one partical
	//figure out the perimeter of the zone in the block
	if(tid==0) {
		_x=partical_cube_id%partical_sidenum;
		_y=(partical_cube_id/partical_sidenum)%partical_sidenum;
		_z=partical_cube_id/(partical_sidenum*partical_sidenum);
		max_x=(_x+1)*partical_r;
		min_x=_x*partical_r;
		max_y=(_y+1)*partical_r;
		min_y=_y*partical_r;
		max_z=(_z+1)*partical_r;
		min_z=_z*partical_r;
		if(min_x-R>0) min_x-=R;	else min_x=0;
		if(min_y-R>0) min_y-=R;	else min_y=0;
		if(min_z-R>0) min_z-=R;	else min_z=0;
		if(max_x+R<1) max_x+=R;	else max_x=1-eps;
		if(max_y+R<1) max_y+=R;	else max_y=1-eps;
		if(max_z+R<1) max_z+=R;	else max_z=1-eps;
		xid=min_x/r,yid=min_y/r,zid=min_z/r;
		Xid=max_x/r,Yid=max_y/r,Zid=max_z/r;
		readnum=(Xid-xid+1)*(Yid-yid+1)*(Zid-zid+1);
	}
	__syncthreads();
	//the perimeter of one block is calculated
	//need syncthreads
	//xid,Xid,yid,Yid,zid,Zid are used for calculate the cube id of the related particals
	//number of cube need to read, and the readnum cubes are stored in shared memory
	//
	//The size of M is the number of related particals, it should be near or little more than
	//the amount the particals of one block
	//
	//The size of BEG is the number of related cube numbers
	//
	__shared__ float4 M[600];
	__shared__ int BEG[600];
	device_MEMSET(readnum,BEG,0,tid,threadnum,blocknum);
	__syncthreads();
	//calculate the location and apartment of vector M
	//for same reason stated above, calculation is only performed on one thread per block
	//_x,_y,_z are used to calculate cube id access
	//BEG vector is used to specify each where to store each related cube in M
	if(tid==0) {
		for(i=0;i<readnum;i++) {
			_z=int(i/((Xid-xid+1)*(Yid-yid+1)))+zid;
			_y=int((i%((Xid-xid+1)*(Yid-yid+1)))/(Xid-xid+1))+yid;
			_x=i%(Xid-xid+1)+xid;
			access=_x+_y*sidenum+_z*sidenum*sidenum;
			BEG[i+1]=BEG[i]+count[access];
		}
		totalnum=BEG[readnum];
	}
	__syncthreads();
	//store related particals in readnum cubes
	//make sure value in count vector is small enough
	step=ceil(float(readnum)/threadnum);
	for(i=tid*step;i<readnum && i<(tid+1)*step;i++) {
		_z=int(i/((Xid-xid+1)*(Yid-yid+1)))+zid;
		_y=int((i%((Xid-xid+1)*(Yid-yid+1)))/(Xid-xid+1))+yid;
		_x=i%(Xid-xid+1)+xid;
		access=_x+_y*sidenum+_z*sidenum*sidenum;
		for(j=0;j<count[access];j++)
			M[BEG[i]+j]=newP[begin[access]+j];
	}
	__syncthreads();
	if(valid) {
		step=ceil(float(totalnum/threadnum));
		for(i=0;i<totalnum;i++) if(
			(M[i].x-_POINT.x)*(M[i].x-_POINT.x)+
			(M[i].y-_POINT.y)*(M[i].y-_POINT.y)+
			(M[i].z-_POINT.z)*(M[i].z-_POINT.z)<=R*R )
			sum+=M[i].w;
		ans[partical_begin[partical_cube_id]+tid]=sum;
	}
	__syncthreads();
//	if(totalnum>T) T=totalnum; //sizeof M
//	if(readnum>RR) RR=readnum; //sizeof BEG
}

//use retrivel vector to ouput ans
__global__ void RESORT(int N,int* partical_retrive,float* ANS,float* ans,int threadnum,int blocknum) {
	int i,j,k;
	int tid,bid,id,totalnum,step;
	tid=threadIdx.x;
	bid=blockIdx.x;
	id=bid*threadnum+tid;
	step=ceil(float(N)/blocknum/threadnum);
	for(i=id*step;i<N && i<(id+1)*step;i++) ANS[partical_retrive[i]]=ans[i];
	__syncthreads();
}
//
int partical_sidenum,partical_cubenum,partical_threadnum;
float partical_r;
//
void N_BODY(int _N, float4* P,int threadnum,int blocknum,int gridnum)
{
	int i,j,k,N=_N;
	float4 *oldP,*newP,*partical_newP;
	float *ans,*ANS;
	int *count,*belong,*retrive,*begin,*partical_count,*partical_belong,*partical_retrive,*partical_begin;
	sidenum=ceil(1.0/r);
	cubenum=sidenum*sidenum*sidenum;
	partical_sidenum=ceil(1.0/partical_r);
	partical_cubenum=partical_sidenum*partical_sidenum*partical_sidenum;
	gridnum=ceil(float(partical_cubenum)/blocknum);
	cudaMalloc((void**)&ans,N*sizeof(float));
	cudaMalloc((void**)&ANS,N*sizeof(float));
	cudaMalloc((void**)&oldP,N*sizeof(float4));
	cudaMalloc((void**)&newP,N*sizeof(float4));
	cudaMalloc((void**)&count,cubenum*sizeof(int));	
	cudaMalloc((void**)&begin,cubenum*sizeof(int));
	cudaMalloc((void**)&belong,N*sizeof(int));
	cudaMalloc((void**)&retrive,N*sizeof(int));
	cudaMalloc((void**)&partical_newP,N*sizeof(float4));
	cudaMalloc((void**)&partical_count,partical_cubenum*sizeof(int));
	cudaMalloc((void**)&partical_begin,partical_cubenum*sizeof(int));
	cudaMalloc((void**)&partical_belong,N*sizeof(int));
	cudaMalloc((void**)&partical_retrive,N*sizeof(int));
	cudaMemcpy(oldP,P,N*sizeof(float4),cudaMemcpyHostToDevice);
	SORT(N,count,begin,belong,retrive,oldP,newP,cubenum,sidenum,r,threadnum,blocknum);
	int a=9;
	SORT(N,partical_count,partical_begin,partical_belong,partical_retrive,oldP,partical_newP,
		partical_cubenum,partical_sidenum,partical_r,threadnum,blocknum);
	//the last parameter is used for the settlement of insufficient shared memory
	//if the shared memory is not enough, we simply cut the particals to several grids
	//each grids could use the total number of shared memory, and the blocks contained 
	//in each grid should be maxized thus the number of grids is minizied
	//successively grids were operately seriesly
	for(i=0;i<gridnum;i++) {
		partical_threadnum=-1;
		for(j=i*blocknum;j<(i+1)*blocknum && j<partical_cubenum;j++) 
			if(partical_count[j]>partical_threadnum) partical_threadnum=partical_count[j];
		CALC<<<blocknum,partical_threadnum>>>(sidenum,partical_threadnum,blocknum,begin,count,ans,newP,
					 i,partical_cubenum,partical_begin,partical_count,R,r,partical_newP,partical_sidenum,partical_r);
	}
	RESORT<<<blocknum,threadnum>>>(N,partical_retrive,ANS,ans,threadnum,blocknum);
	cudaMemcpy(OUT,ANS,N*sizeof(float),cudaMemcpyDeviceToHost);
	for(i=0;i<N;i++) printf("%f\n",OUT[i]);
} 
void input() {
	int i,j,k;
	FILE* file=fopen("input.txt","r");
	fscanf(file,"%d%f",&N,&R);
	for(i=0;i<N;i++) fscanf(file,"%f%f%f%f",&(P[i].x),&(P[i].y),&(P[i].z),&(P[i].w));
}
void set_threadnum_blocknum() {
	threadnum=partical_threadnum=128;
	blocknum=2;
	r=0.01;
	partical_r=0.05;
}
int main(int argc, char** argv)
{
	freopen("output.txt","w",stdout);
	if(!InitCUDA()) return 0;
//	printf("CUDA initialized.\n");	
	input();
	set_threadnum_blocknum();
	N_BODY(N,P,threadnum,blocknum,gridnum);
//	printf("%d %d\naa",T,RR);
	return 0;
}