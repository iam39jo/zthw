// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
void randomInit(float* data, int size);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len, const unsigned int dimension);
bool hComparef( const float* reference, const float* data, const unsigned int len);

// includes, kernels
#include <knn_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	printf("particle num is:%d \n",particle_num);
    runTest( argc, argv);

    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//runTest
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

	srand((unsigned int)time(0));

	unsigned int i_data_size = particle_num * (particle_dimension + 1);		//产生粒子数所需数组大小 ： 粒子数×（粒子维数＋1）
	unsigned int i_mem_size = sizeof( float) * i_data_size;					//用于存储粒子数据所需空间字节数				
	unsigned int o_mem_size = sizeof( float) * particle_num;				//用于存储结果所需空间字节数

    unsigned int timer = 0;
	cutilCheckError( cutCreateTimer( &timer));
	cutilCheckError( cutStartTimer( timer));

    // 分配 host memory
    float* h_idata = (float*) malloc( i_mem_size);		//分配数据空间在h_idata上
    
    randomInit(h_idata, i_data_size);					//产生计算数据放在h_idata


    // 分配device memory
    float* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, i_mem_size));

    // 把内存中的h_idata拷贝到显存中的d_idata中去
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, i_mem_size,
                                cudaMemcpyHostToDevice) );

    // 分配 device memory 用于存储结果
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, o_mem_size));

    // 配置内核Grid和Block都是一维的
    dim3  grid( block_num, 1, 1);
    dim3  threads( thread_num, 1, 1);

    // 执行kernel
    testKernel<<< grid, threads >>>( d_idata, d_odata);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // 分配内存空间存储device上计算的结果
    float* h_odata = (float*) malloc( o_mem_size);
    
	// 把device上的结果拷贝到host上的h_odata数组中去
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, o_mem_size,
                                cudaMemcpyDeviceToHost) );


	cutilCheckError( cutStopTimer( timer));
	printf( "Processing time in GPU: %f (ms)\n", cutGetTimerValue( timer));

	/////////////////////////////////////////////////////////////////////////////
	//////CPU上的计算
	////////////////////////////////////////////////////////////////////////////
    cutilCheckError( cutStartTimer( timer));
	
	// 分配内存空间用于存储CPU上的计算结果
    float* reference = (float*) malloc( o_mem_size);

    computeGold( reference, h_idata, particle_num, particle_dimension);

	cutilCheckError( cutStopTimer( timer));
    printf( "Processing time in CPU: %f (ms)\n", cutGetTimerValue( timer));
    
	cutilCheckError( cutDeleteTimer( timer));

    bool res = hComparef( reference, h_odata, particle_num);
    printf( "\n GPU上与CPU上测试结果是否通过： %s\n", (1 == res) ? "PASSED" : "FAILED");

    // 回收内存空间
    free( h_idata);
    free( h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();
}

//产生随机数据函数
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
      data[i] = rand() / (float)RAND_MAX;
}

//比较测试结果，输出错误个数
bool hComparef( const float* reference, const float* data, const unsigned int len)
{        
        bool result = true;
		int error_num;
		error_num = 0;

        for( unsigned int i = 0; i < len; ++i) {

            float diff = reference[i] - data[i];
            bool comp = (diff <= 0.0f) && (diff >= -0.0f);
            result &= comp;

			if( ! comp) 
				error_num++;
        }
		printf("ERROR_NUM = %d ,\n", error_num);
        return (result) ? true : false;
}
