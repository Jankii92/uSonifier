#include "cuda.h"
#include "improc.h"
#include <stdio.h>
#include <chrono>

#include <iostream>

#define PI 3.14159265
#define WIDTH	640
#define HEIGHT	480


using namespace std;


int calcSum(unsigned char *src){
	
	int xh = 480/2;
	int yh = 640/2;
	int sum = 0;
	for( int x = xh-50; x < xh+50 ; x++){
		for( int y = yh-50; y < yh+50 ; y++){
			sum+=src[y*640+x];
		}
	}
	return sum;


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
	blur_noShare_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, k, gpudataSrc, gpudataOut );
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
void cv::gpu::mj::sobel(const int rows,const int cols, const unsigned char *src, unsigned char* dst, int mode){
	
	int N = WIDTH;
	int M = HEIGHT;
	
	unsigned char* gpudataSrc;
	
	unsigned char* gpudataMid;
	unsigned char* gpudataOut;
	
	const int size = sizeof(unsigned char)*rows*cols;
		
	
	cudaMalloc((void **)&gpudataSrc, size);
	cudaMalloc((void **)&gpudataOut, size);
	cudaMalloc((void **)&gpudataMid, size);
	
	cudaMemcpyAsync(gpudataSrc, src, size, cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(32,32);
	dim3 numbBlocks(N/ threadsPerBlock.x,M/ threadsPerBlock.y); 
	
	cout<<"MAT!!!!!!!"<<endl;

	float angle = -2.0f;
	rotate_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, gpudataSrc, gpudataMid, angle*PI/180);
	sobel_abs_GPU<<<numbBlocks, threadsPerBlock>>>(rows,cols, gpudataMid, gpudataOut, 1);
	cudaMemcpyAsync(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	cout<<"Angle:"<<angle<<" Sum:"<<calcSum(dst)<<endl;
	angle = -1.5f;
	rotate_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, gpudataSrc, gpudataMid, angle*PI/180);
	sobel_abs_GPU<<<numbBlocks, threadsPerBlock>>>(rows,cols, gpudataMid, gpudataOut, 1);
	cudaMemcpyAsync(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	cout<<"Angle:"<<angle<<" Sum:"<<calcSum(dst)<<endl;
	angle = 1.0f;
	rotate_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, gpudataSrc, gpudataMid, angle*PI/180);
	sobel_abs_GPU<<<numbBlocks, threadsPerBlock>>>(rows,cols, gpudataMid, gpudataOut, 1);
	cudaMemcpyAsync(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	cout<<"Angle:"<<angle<<" Sum:"<<calcSum(dst)<<endl;
	angle = 0.5f;
	rotate_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, gpudataSrc, gpudataMid, angle*PI/180);
	sobel_abs_GPU<<<numbBlocks, threadsPerBlock>>>(rows,cols, gpudataMid, gpudataOut, 1);
	cudaMemcpyAsync(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	cout<<"Angle:"<<angle<<" Sum:"<<calcSum(dst)<<endl;
	angle = 0.0f;
	rotate_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, gpudataSrc, gpudataMid, angle*PI/180);
	sobel_abs_GPU<<<numbBlocks, threadsPerBlock>>>(rows,cols, gpudataMid, gpudataOut, 1);
	cudaMemcpyAsync(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	cout<<"Angle:"<<angle<<" Sum:"<<calcSum(dst)<<endl;
	angle = 0.5f;
	rotate_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, gpudataSrc, gpudataMid, angle*PI/180);
	sobel_abs_GPU<<<numbBlocks, threadsPerBlock>>>(rows,cols, gpudataMid, gpudataOut, 1);
	cudaMemcpyAsync(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	cout<<"Angle:"<<angle<<" Sum:"<<calcSum(dst)<<endl;
	angle = 0.0f;
	rotate_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, gpudataSrc, gpudataMid, angle*PI/180);
	sobel_abs_GPU<<<numbBlocks, threadsPerBlock>>>(rows,cols, gpudataMid, gpudataOut, 1);
	cudaMemcpyAsync(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	cout<<"Angle:"<<angle<<" Sum:"<<calcSum(dst)<<endl;
	
	cudaFree(gpudataMid);
	cudaFree(gpudataSrc);
	cudaFree(gpudataOut);
}

void cv::gpu::mj::rectif(const int rows,const int cols, const unsigned char *srcL, const unsigned char *srcR, unsigned char *dstL, unsigned char *dstR, unsigned char * out){
	
	int N = WIDTH;
	int M = HEIGHT;
	
	unsigned char* gpudataSrcL;
	unsigned char* gpudataSrcR;
	unsigned char* gpudataOutL;
	unsigned char* gpudataOutR;
	unsigned char* gpudataOut;
		
	const int size = sizeof(unsigned char)*rows*cols;
		
	
	cudaMalloc((void **)&gpudataSrcL, size);
	cudaMalloc((void **)&gpudataOutL, size);
	cudaMalloc((void **)&gpudataSrcR, size);
	cudaMalloc((void **)&gpudataOutR, size);
	cudaMalloc((void **)&gpudataOut, size);
	
	cudaMemcpy(gpudataSrcL, srcL, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpudataSrcR, srcR, size, cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(32,32);
	dim3 numbBlocks(N/ threadsPerBlock.x,M/ threadsPerBlock.y); 
	
	//cout<<"MAT!!!!!!!"<<endl;

	float angle = 0.0f;
	rotate_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, gpudataSrcR, gpudataOutR, 0);
	rotate_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, gpudataSrcL, gpudataOutL, angle*PI/180);
	cudaMemcpyAsync(dstL, gpudataOutL, size, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(dstR, gpudataOutR, size, cudaMemcpyDeviceToHost);

	blend_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols,  gpudataOutL,  gpudataOutR, gpudataOut, 0.5f);
	
	cudaMemcpy(out, gpudataOut, size, cudaMemcpyDeviceToHost);
	cudaFree(gpudataSrcL);
	cudaFree(gpudataOutL);
	cudaFree(gpudataSrcR);
	cudaFree(gpudataOutR);
	cudaFree(gpudataOut);
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
























