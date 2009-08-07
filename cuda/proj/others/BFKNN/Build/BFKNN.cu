#include <komrade/device_vector.h>
#include <komrade/transform.h>
#include <komrade/range.h>
#include <komrade/copy.h>
#include <komrade/fill.h>
#include <komrade/sort.h>
#include <komrade/replace.h>
#include <komrade/functional.h>
#include <iostream>
#include <iterator>
#include <math.h>
#include <memory>
#include <boost/timer.hpp>

#include <vector>

using namespace std;

#include <ANN/ANN.h>
#pragma comment(lib,"ANN.lib")

struct sqrtdist
{
	sqrtdist(const float qx, const float qy, const float qz) : x(qx), y(qy), z(qz)
	{
	}
	float x,y,z;
	
	__host__ __device__ float operator()(const float3& p) const
	{
		float a = p.x - x;
		float b = p.y - y;
		float c = p.z - z;
		return a*a + b*b + c*c;
	}
};

//void bfknn(const float3 lp, const std::vector<float3>& p, std::vector<int>& idx, std::vector<float>& dist)
//{
//	//copy data from host to device
//	komrade::device_vector<float3> pdev(p.begin(), p.end());
//	komrade::device_vector<int> idxdev(p.size());
//	komrade::device_vector<float> distdev(p.size());
//	//initialize the indices
//	for(size_t i=0; i<p.size(); ++i)
//	{
//		idxdev[i] = i;
//	}
//	//brust force sort the point set	
//	komrade::transform(pdev.begin(), pdev.end(), distdev.begin(), sqrtdist(lp.x, lp.y, lp.z));
//	komrade::sort_by_key(distdev.begin(), distdev.end(), idxdev.begin());
//	
//	//readback to host
//	idx.resize(idxdev.size());
//	dist.resize(distdev.size());
//	komrade::copy(idxdev.begin(), idxdev.end(), idx.begin());
//	komrade::copy(distdev.begin(), distdev.end(), dist.begin());
//}

int main(int argc, char* argv[])
{
	if( argc < 2 )
		return -1;
			
	int N = 0;
	sscanf(argv[1],"%d",&N);
	boost::timer Stopwatch;
	ANNpointArray ANNPh = annAllocPts(N,3);
	komrade::host_vector<float3> Ph(N);
	komrade::device_vector<int> Idx(N);
	komrade::device_vector<float> Dist(N);

	for(int i=0; i<N; ++i)
	{
		float x = (float)rand() / (float)RAND_MAX;;
		float y = (float)rand() / (float)RAND_MAX;;
		float z = (float)rand() / (float)RAND_MAX;;
		
		Ph[i].x = x;
		Ph[i].y = y;
		Ph[i].z = z;
		
		ANNPh[i][0] = x;
		ANNPh[i][1] = y;
		ANNPh[i][2] = z;
				
		Idx[i] = i;
	}
	
	
	cout<<fixed<<"Generate Data Used "<<Stopwatch.elapsed()<<endl;
	Stopwatch.restart();
		
	float QueryPos[3];
	QueryPos[0] = (float)rand() / (float)RAND_MAX * 0.1f + 0.5f;
	QueryPos[1] = (float)rand() / (float)RAND_MAX * 0.1f + 0.5f;
	QueryPos[2] = (float)rand() / (float)RAND_MAX * 0.1f + 0.5f;
		
	komrade::device_vector<float3> Pd = Ph;
	cout<<fixed<<"Copy Data Used "<<Stopwatch.elapsed()<<endl;
	Stopwatch.restart();
	
	komrade::transform(Pd.begin(), Pd.end(), Dist.begin(), sqrtdist(QueryPos[0], QueryPos[1], QueryPos[2]));
	komrade::sort_by_key(Dist.begin(), Dist.end(), Idx.begin());
	cout<<fixed<<"Sort Data by CUDA Used "<<Stopwatch.elapsed()<<endl;
	Stopwatch.restart();
	
	ANNkd_tree ANNTree(ANNPh,N,3);
	ANNidx ANNIdx[1];
	ANNdist ANNDst[1];
	ANNTree.annkSearch(QueryPos,1,ANNIdx,ANNDst);
	cout<<fixed<<"Sort Data by ANN Used "<<Stopwatch.elapsed()<<endl;
	
	return 0;
}