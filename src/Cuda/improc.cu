#include "improc.h"


__global__ void blur_GPU(const int rows,const  int cols, const int k, const unsigned char *img, unsigned char *result){

	__shared__ unsigned char s[1024];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int t = y*cols+x; 
    s[threadIdx.y*32+threadIdx.x] = img[t]; 
    const int r = k/2;
    
    if(x <= r || x >= cols-r || y <= r || y >= rows-r ) return;
	long sum = 0;
	int xx;
	int yy;
	int rr = k*k;
	
    if(threadIdx.x < r || threadIdx.x >= 32-r || threadIdx.y < r || threadIdx.y >= 32-r){
    	xx = x-r;
	do{
		yy = y-r;
		do{
			sum+=img[yy*cols+xx];
		}while(yy++ < y+r);
	}while(xx++ < x+r);
	result[t] = (unsigned int)(sum/rr);	
	return;
		
	}
	__syncthreads();
		 
	xx = threadIdx.x-r;
	do{
		yy = threadIdx.y-r;
		do{
			sum+=s[yy*32+xx];
		}while(yy++ < threadIdx.y+r);
	}while(xx++ < threadIdx.x+r);
	result[t] = (unsigned int)(sum/rr);	
}

__global__ void blur_noShare_GPU( const int rows, const int cols,const int k, const unsigned char *img, unsigned char *result){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int t = y*cols+x; 
    const int r = k/2;
    
    if(x <= r || x >= cols-r || y <= r || y >= rows-r) return;
	
	long sum = 0;
    int xx;
    int yy;
    int rr = k*k;
     
	xx = x-r;
	do{
		yy = y-r;
		do{
			sum+=img[yy*cols+xx];
		}while(yy++ < y+r);
	}while(xx++ < x+r);
	result[t] = (unsigned int)(sum/rr);	
}
