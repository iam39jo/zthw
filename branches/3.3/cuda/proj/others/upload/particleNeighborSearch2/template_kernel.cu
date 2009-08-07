/*
* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.  Users and possessors of this source code
* are hereby granted a nonexclusive, royalty-free license to use this code
* in individual and commercial software.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Device code.
*/

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#define thread_num			256
#define block_num			64

#define	particle_num		10240
#define	particle_dimension	3
#define	data_dimension		4
#define radius_square 		((float)(1 * 1))

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, float* g_odata) 
{
	const int step = gridDim.x * blockDim.x;
	int base_index = blockIdx.x * blockDim.x + threadIdx.x;
	float value_sum;
	float square;
	float axis_diff;

	unsigned int dim_i;
	
	float obj_data[particle_dimension];

	for(;base_index < particle_num; base_index += step)
	{
		value_sum = 0;

		memcpy(obj_data, g_idata+base_index*(particle_dimension+1), sizeof(float)*particle_dimension);

		__shared__ float sdata[thread_num*data_dimension];

		for(int g=0;g<particle_num/thread_num;++g)
		{
			memcpy(&sdata[threadIdx.x*data_dimension], &g_idata[data_dimension*thread_num*g+threadIdx.x*data_dimension], data_dimension*sizeof(float));

			__syncthreads();

			for(int e=0;e<thread_num;++e)			
			{	
				square = 0;

				for(dim_i=0;dim_i<particle_dimension;++dim_i)
				{
					axis_diff = obj_data[dim_i]-sdata[e*4+dim_i];
					square += axis_diff * axis_diff;
				}
				if(square - radius_square < 1.0/1024)
				{
					value_sum += sdata[e*(particle_dimension+1) + particle_dimension];
				}
			}
			__syncthreads();
		}

		g_odata[base_index]=value_sum;
	}
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
