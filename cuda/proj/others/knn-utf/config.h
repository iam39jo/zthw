#ifndef _CONFIG_H_
#define _CONFIG_H_

// Thread block size
#define BLOCK_SIZE 16

#define thread_num			256					//每个block线程数256
#define block_num			64					//block的维数（目前程序仅采用1维）64

#define	particle_num		10240				//为例子总数:256,1240,12400,124000
#define	particle_dimension	3					//粒子坐标 均为0～1之间浮点随机数
#define	data_dimension		4					//数据的属性值 均为0～1之间浮点随机数
#define radius_square 		((float)(1 * 1))	//为粒子搜索半径

#endif // _CONFIG_H_