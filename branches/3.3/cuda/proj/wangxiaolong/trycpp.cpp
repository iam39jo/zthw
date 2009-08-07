#include<iostream>
using namespace std;
#include<stdio.h>
#include<stdlib.h>
const int MAX=10000;
float R;
int N;
float p[MAX][4];
int main()
{
	int i,j,k;
	freopen("out1.out","w",stdout);
	FILE* file=fopen("input.txt","r");
	fscanf(file,"%d%f",&N,&R);
	for(i=0;i<N;i++) 
		fscanf(file,"%f%f%f%f",&p[i][0],&p[i][1],&p[i][2],&p[i][3]);
	for(i=0;i<N;i++) {
		float sum=0;
		for(j=0;j<N;j++) 
		if( (p[i][0]-p[j][0])*(p[i][0]-p[j][0])+
			(p[i][1]-p[j][1])*(p[i][1]-p[j][1])+
			(p[i][2]-p[j][2])*(p[i][2]-p[j][2])<=R*R)
			sum+=p[j][3];
		printf("%f\n",sum);
	}
}
