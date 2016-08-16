#include "cuda.h"
#include <stdio.h>
#include <chrono>

#include <iostream>


#define WIDTH	640
#define HEIGHT	480


using namespace std;


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

__global__ void blur_noShere_GPU( const int rows, const int cols,const int k, const unsigned char *img, unsigned char *result){
	
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


void cv::gpu::mj::test(){
	printf("lallalala\n");
	
}

void cv::gpu::mj::blur(const int rows,const int cols, const int k, const unsigned char *src, unsigned char* dst){
	
	auto start1 = std::chrono::system_clock::now();
	int N = WIDTH;
	int M = HEIGHT;
	//cudaSetDeviceFlags(cudaDeviceMapHost);
	
	unsigned char* gpudataSrc;
	unsigned char* gpudataOut;
	unsigned char* cpudataOut;
	
	int size = sizeof(unsigned char)*rows*cols;
	
	cudaMallocHost 	((void **)&cpudataOut,size);	
	
	auto stop1 = std::chrono::system_clock::now();
	cudaMalloc((void **)&gpudataSrc, size);
	cudaMalloc((void **)&gpudataOut, size);
	//cudaHostGetDevicePointer((void **)&gpudataOut,  (void *) dst , 0);
	cudaMemcpyAsync(gpudataSrc, src, size, cudaMemcpyHostToDevice);
	
	auto stop2 = std::chrono::system_clock::now();
	dim3 threadsPerBlock(32,32);
	dim3 numbBlocks(N/ threadsPerBlock.x,M/ threadsPerBlock.y); 
	//auto start1 = std::chrono::system_clock::now();
	auto stop3 = std::chrono::system_clock::now();
	blur_noShere_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, k, gpudataSrc, gpudataOut );
	blur_noShere_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, k, gpudataOut, gpudataSrc );
	blur_noShere_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, k, gpudataSrc, gpudataOut );
	blur_noShere_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, k, gpudataOut, gpudataSrc );
	blur_noShere_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, k, gpudataSrc, gpudataOut );
	//blur_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, k, gpudataSrc, gpudataOut);
	auto stop4 = std::chrono::system_clock::now();
	cudaMemcpyAsync(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	auto stop5 = std::chrono::system_clock::now();
	cudaFree(gpudataSrc);
	cudaFree(gpudataOut);
	auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
	auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - stop1);
	auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop3 - stop2);
	auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(stop4 - stop3);
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop5 - start1);
	
    //cout<<(int)duration1.count()<<" "<<(int)duration2.count()<<" " <<(int)duration3.count()<<" " <<(int)duration4.count()<<" " <<(int)duration.count()<< endl;

	
}

void cv::gpu::mj::realocHostMem(int sizec, unsigned char *img){
	unsigned char* cpudataSrc;
	cudaMallocHost 	((void **)&cpudataSrc,sizeof(unsigned char));
	//cudaMemcpy(cpudataSrc, img, sizec, cudaMemcpyHostToHost);
	//img = cpudataSrc;		
}

void cv::gpu::mj::cudaMemAlocImagePtr(unsigned char *dest, int size){
	//cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaMallocHost 	((void **)&dest, size);
	//cudaHostAlloc((void **)&dest,  size,  cudaHostAllocMapped);
}

void cv::gpu::mj::cudaMemcpyHtoH(unsigned char *src, unsigned char *dest, int size){
	//cudaMallocHost 	((void **)&dest,size);
	cudaMemcpy(dest, src, size, cudaMemcpyHostToHost);	
}






















