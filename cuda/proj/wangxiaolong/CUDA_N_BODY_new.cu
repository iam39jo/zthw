#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
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
	__syncthreads();
}
//calculate the cube id of each partical
//N=gridnum*blocknum*threadnum
//run 1 times, with blocknum blocks, threadnum threads
__global__ void SORT_1(int N,int* count,int* begin,int* belong,
					   int* retrive,float4* oldP,float4* newP,int sidenum,float r,int threadnum,int blocknum) {
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
__global__ void SORT_2(int N,int* count,int* begin,int* belong,
					   int* retrive,float4* oldP,float4* newP,int cubenum,int threadnum,int blocknum) {
	int i,j,k,tid,bid,id;
	if(threadIdx.x==0 && blockIdx.x==0) 
		for(i=1;i<cubenum;i++) begin[i]+=begin[i-1];
}
//place partical to new vector newP, and set retrival vector for future output
//run 1 times, with blocknum blocks, threadnum threads
__global__ void SORT_3(int N,int* count,int* begin,int* belong,
					   int* retrive,float4* oldP,float4* newP,int threadnum,int blocknum) {
	int i,j,k,tid,bid,id,beg,end,step,ID;
	tid=threadIdx.x;
	bid=blockIdx.x;
	id=tid+bid*threadnum;
	step=ceil(float(N)/blocknum/threadnum);
	beg=step*id;
	end=beg+step;
	for(i=beg;i<N && i<end;i++) { 
		ID=belong[i]; j=--begin[ID]; retrive[j]=i; newP[j]=oldP[i];
	}
}
void SORT(int N,int* count,int* begin,int* belong,
					   int* retrive,float4* oldP,float4* newP,int threadnum,int blocknum) {
	MEMSET<<<blocknum,threadnum>>>(cubenum,begin,0,threadnum,blocknum);
	MEMSET<<<blocknum,threadnum>>>(cubenum,count,0,threadnum,blocknum);
	SORT_1<<<blocknum,threadnum>>>(N,count,begin,belong,retrive,oldP,newP,sidenum,r,threadnum,blocknum);
	//only need to run on 1 thread
	SORT_2<<<1,1>>>(N,count,begin,belong,retrive,oldP,newP,cubenum,threadnum,blocknum);
	SORT_3<<<blocknum,threadnum>>>(N,count,begin,belong,retrive,oldP,newP,threadnum,blocknum);
}
//make sure that threadnum*blocknum*gridnum equals to the number of total particals
//this kernel run with a gridnum times for sentence 
//increase the gridnum value to increase the memory limit for each threadnum
//each thread deals with only one partical
__global__ void CALC(float4* newP,float* ans,int* count,int* begin,int sidenum,int N,float R,float r,
					 int threadnum,int blocknum,int gridnum,int gid) {
	float eps=0.00002;
	int i,j,k,tid,bid,id,beg,end,step;
	__shared__ float max_x,max_y,max_z,min_x,min_y,min_z;
	float sum=0;
	__shared__ int xid,Xid,yid,Yid,zid,Zid,readnum,totalnum;
	int access,_x,_y,_z;
	float4 _POINT;
	tid=threadIdx.x;
	bid=blockIdx.x;
	id=gid*blocknum*threadnum+bid*threadnum+tid;
	max_x=max_y=max_z=-1, min_x=min_y=min_z=1;
	__syncthreads();
	//every threads calculate one partical
	//figure out the perimeter of the zone in the block
	_POINT=newP[id];
	if(id>=N && max_x<_POINT.x) max_x=_POINT.x;
	if(id>=N && max_y<_POINT.y) max_y=_POINT.y;
	if(id>=N && max_z<_POINT.z) max_z=_POINT.z;
	if(id>=N && min_x>_POINT.x) min_x=_POINT.x;
	if(id>=N && min_y>_POINT.y) min_y=_POINT.y;
	if(id>=N && min_z>_POINT.z) min_z=_POINT.z;
	if(id>=N && min_x-R>0) min_x-=R;	else min_x=0;
	if(id>=N && min_y-R>0) min_y-=R;	else min_y=0;
	if(id>=N && min_z-R>0) min_z-=R;	else min_z=0;
	if(id>=N && max_x+R<1) max_x+=R;	else max_x=1-eps;
	if(id>=N && max_y+R<1) max_y+=R;	else max_y=1-eps;
	if(id>=N && max_z+R<1) max_z+=R;	else max_z=1-eps;
	__syncthreads();
	xid=min_x/r,yid=min_y/r,zid=min_z/r;
	Xid=max_x/r,Yid=max_y/r,Zid=max_z/r;
	//the perimeter of one block is calculated
	//need syncthreads
	//xid,Xid,yid,Yid,zid,Zid are used for calculate the cube id of the related particals
	//number of cube need to read, and the readnum cubes are stored in shared memory
	readnum=(Xid-xid+1)*(Yid-yid+1)*(Zid-zid+1);
	__shared__ float4 M[1000];
	__shared__ int BEG[MAX];
	device_MEMSET(readnum,BEG,0,tid,threadnum,blocknum);
	//calculate the location and apartment of vector M
	//for same reason stated above, calculation is only performed on one thread per block
	//_x,_y,_z are used to calculate cube id access
	//BEG vector is used to specify each where to store each related cube in M
	if(id<N && tid==0) 	{
		for(i=0;i<readnum;i++) {
			_z=int(i/(Xid-xid+1)/(Yid-yid+1))+xid;
			_y=int((i-_z*(Xid-xid+1)*(Yid-yid+1))/(Xid-xid+1))+yid;
			_x=i%(Xid-xid+1)+xid;
			access=_x+_y*sidenum+_z*sidenum*sidenum;
			BEG[i+1]=BEG[i]+count[access];
		}
		totalnum=BEG[readnum];
	}
	__syncthreads();
	//store related particals in readnum cubes
	//make sure value in count vector is small enough
	if(id<N) step=ceil(float(readnum)/threadnum);
	if(id<N) for(i=tid*step;i<readnum && i<(tid+1)*step;i++) {
		_z=int(i/(Xid-xid+1)/(Yid-yid+1))+xid;
		_y=int((i-_z*(Xid-xid+1)*(Yid-yid+1))/(Xid-xid+1))+yid;
		_x=i%(Xid-xid+1)+xid;
		access=_x+_y*sidenum+_z*sidenum*sidenum;
		for(j=0;j<count[access];j++) M[BEG[i]+j]=newP[begin[access]+j];
	}
	__syncthreads();
	if(id<N) step=ceil(float(totalnum/threadnum));
	if(id<N) for(i=0;i<totalnum;i++) if(
		(M[i].x-_POINT.x)*(M[i].x-_POINT.x)+
		(M[i].y-_POINT.y)*(M[i].y-_POINT.y)+
		(M[i].z-_POINT.z)*(M[i].z-_POINT.z)<=R*R )
		sum+=M[i].w;
	if(id<N) ans[id]=sum;
	__syncthreads();
}
//use retrivel vector to ouput ans
__global__ void RESORT(int N,int* retrive,float* ANS,float* ans,int threadnum,int blocknum) {
	int i,j,k;
	int tid,bid,id,totalnum,step;
	tid=threadIdx.x;
	bid=blockIdx.x;
	id=bid*threadnum+tid;
	step=ceil(float(N)/blocknum/threadnum);
	for(i=id*step;i<N && i<(id+1)*step;i++) 
		ANS[retrive[i]]=ans[i];
	__syncthreads();
}
void N_BODY(int _N, const float4* P,int threadnum,int blocknum,int gridnum)
{
	int i,j,k,N=_N;
	float4 *oldP,*newP;
	float *ans,*ANS;
	int *count,*belong,*retrive,*begin;
	sidenum=ceil(1.0/r);
	cubenum=sidenum*sidenum*sidenum;
	cudaMalloc((void**)&ans,N*sizeof(float));
	cudaMalloc((void**)&ANS,N*sizeof(float));
	cudaMalloc((void**)&oldP,N*sizeof(float4));
	cudaMalloc((void**)&newP,N*sizeof(float4));
	cudaMalloc((void**)&count,cubenum*sizeof(int));	
	cudaMalloc((void**)&begin,cubenum*sizeof(int));
	cudaMalloc((void**)&belong,N*sizeof(int));
	cudaMalloc((void**)&retrive,N*sizeof(int));
	cudaMemcpy(oldP,P,N*sizeof(float4),cudaMemcpyHostToDevice);
	SORT(N,count,begin,belong,retrive,oldP,newP,threadnum,blocknum);
	//the last parameter is used for the settlement of insufficient shared memory
	//if the shared memory is not enough, we simply cut the particals to several grids
	//each grids could use the total number of shared memory, and the blocks contained 
	//in each grid should be maxized thus the number of grids is minizied
	//successively grids were operately seriesly
	for(i=0;i<gridnum;i++) CALC<<<blocknum,threadnum>>>(newP,ans,count,begin,sidenum,N,R,r,threadnum,blocknum,gridnum,i);
	RESORT<<<blocknum,threadnum>>>(N,retrive,ANS,ans,threadnum,blocknum);
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
	threadnum=64;
	blocknum=2;
	r=0.1;
	gridnum=ceil(float(N)/threadnum/blocknum);
}
int main(int argc, char** argv)
{
	freopen("output.txt","w",stdout);
	if(!InitCUDA()) return 0;
//	printf("CUDA initialized.\n");
	input();
	set_threadnum_blocknum();
	N_BODY(N,P,threadnum,blocknum,gridnum);
	return 0;
}