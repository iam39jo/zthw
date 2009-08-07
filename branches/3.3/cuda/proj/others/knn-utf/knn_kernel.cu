#ifndef _KNN_KERNEL_H_
#define _KNN_KERNEL_H_

#include "config.h"

////////////////////////////////////////////////////////////////////////////////
//! 计算粒子距离并把小于距离平方小于radius_square的粒子所带浮点数相加
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, float* g_odata) 
{
	const int step = gridDim.x * blockDim.x;					//控制循环步长
	int base_index = blockIdx.x * blockDim.x + threadIdx.x;		//所求粒子的序号
	float value_sum;											//浮点数相加之各
	float square;												//粒子之间距离的平方
	float axis_diff;											//粒子之间坐标之差

	unsigned int dim_i;
	
	float obj_data[particle_dimension];

	for(;base_index < particle_num; base_index += step)
	{
		value_sum = 0;

		//把globle memory里的一个粒子的信息拷贝到obj_data上去，用于计算该点与其它粒子的距离
		for(int i=0; i<particle_dimension; i++)
			obj_data[i] = g_idata[base_index*data_dimension+i];
		
		//分配share memory胜于存放所有点信息，这里只能存放thread_num个点，需要particle_num/thread_num个线程调入
		__shared__ float sdata[thread_num*data_dimension];

		for(int g=0;g<particle_num/thread_num;++g)
		{
			//把粒子信息拷贝到share memory里去
			for(int i=0; i<data_dimension; i++)
				sdata[threadIdx.x*data_dimension+i] = g_idata[data_dimension*thread_num*g+threadIdx.x*data_dimension+i];

			__syncthreads();

			for(int e=0;e<thread_num;++e)			
			{	
				square = 0;

				//计算粒子与其它粒子之间距离
				for(dim_i=0;dim_i<particle_dimension;++dim_i)
				{
					axis_diff = obj_data[dim_i]-sdata[e*4+dim_i];
					square += axis_diff * axis_diff;
				}

				//比较距离
				if(square < radius_square)
				{
					value_sum += sdata[e*(particle_dimension+1) + particle_dimension];//所带浮点数相加
				}
			}
			__syncthreads();
		}

		g_odata[base_index]=value_sum;//把信息放到目标globe memory里的数组中去
	}
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
