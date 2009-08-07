#include<stdio.h>
#include "config.h"
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len, const unsigned int dimension);

////////////////////////////////////////////////////////////////////////////////
//! 计算与粒子距离小于radius的粒子第四个数据之和
//! Each element is multiplied with the number of threads / array length
//! @param reference  用于存放结果
//! @param idata      输入的数据
//! @param len        粒子的总数
//! @param dimension  存放粒子坐标的维数 
////////////////////////////////////////////////////////////////////////////////
void
computeGold( float* reference, float* idata, const unsigned int len, const unsigned int dimension) 
{
	//const float radius_square = (float)(1 * 1);
	float value_sum;
	float square;
	float axis_diff;

	for(unsigned int base_index=0; base_index<len; base_index++)
	{
		value_sum = 0;

		for(unsigned int dest_index=0; dest_index<len; dest_index++)
		{
			square = 0;

			//计算两点之间的距离的平方即x*x+y*y+z*z
			for(unsigned int i=0; i<dimension; i++)
			{
				axis_diff = idata[base_index*(dimension+1)+i]-idata[dest_index*(dimension+1)+i];
				square += axis_diff * axis_diff;
			}

			//if(square - radius_square < 1.0/1024) 
			if(square < radius_square)
			{
				value_sum += idata[dest_index*(dimension+1) + dimension];
			}
		}
		reference[base_index]=value_sum;

	}
}

