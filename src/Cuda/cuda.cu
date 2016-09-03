#include "cuda.h"
#include "improc.h"
#include <stdio.h>
#include <chrono>

#include <iostream>

#define PI 3.14159265
#define NUM_STREAMS 2
#define WIDTH	640
#define HEIGHT	480


using namespace std;



void cv::gpu::mj::blur(const int rows,const int cols, const int k, unsigned char *src, unsigned char* dst){
	
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
	cudaMemcpy(gpudataSrc, src, size, cudaMemcpyHostToDevice);
	
	auto stop2 = std::chrono::system_clock::now();
	dim3 threadsPerBlock(16,16);
	dim3 numbBlocks(N/ threadsPerBlock.x,M/ threadsPerBlock.y); 
	//auto start1 = std::chrono::system_clock::now();
	//cudaDeviceSynchronize();
	auto stop3 = std::chrono::system_clock::now();
	//blur_noShare_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, k, gpudataSrc, gpudataOut );
	//cudaDeviceSynchronize();
	auto stop4 = std::chrono::system_clock::now();
	//cudaDeviceSynchronize();
	cout<<"1: Done!!!"<<endl;
	//blur_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, k, gpudataSrc, gpudataOut);
	cout<<"3: Done!!!"<<endl;
	//cudaDeviceSynchronize();
	auto stop5 = std::chrono::system_clock::now();
	cudaMemcpy(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	cudaFree(gpudataSrc);
	cudaFree(gpudataOut);
	auto duration1 = (std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count());
	auto duration2 = (std::chrono::duration_cast<std::chrono::microseconds>(stop2 - stop1).count());
	auto duration3 = (std::chrono::duration_cast<std::chrono::microseconds>(stop3 - stop2).count());
	auto duration4 = (std::chrono::duration_cast<std::chrono::microseconds>(stop4 - stop3).count());
	auto duration5 = (std::chrono::duration_cast<std::chrono::microseconds>(stop5 - stop4).count());
	auto duration =(std::chrono::duration_cast<std::chrono::microseconds>(stop5 - start1).count());
	
    //cout<<(int)duration1<<" "<<(int)duration2<<" " <<(int)duration3<<" " <<(int)duration4<<" " <<(int)duration5<<(int)duration6<< endl;
}
void cv::gpu::mj::sobel(const int rows,const int cols, unsigned char *src, unsigned char* dst, int mode){
	
	int N = WIDTH;
	int M = HEIGHT;
	
	unsigned char* gpudataSrc;
	
	unsigned char* gpudataOut;
	
	const int size = sizeof(unsigned char)*rows*cols;
		
	
	cudaMalloc((void **)&gpudataSrc, size);
	cudaMalloc((void **)&gpudataOut, size);
	
	cudaMemcpyAsync(gpudataSrc, src, size, cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(32,32);
	dim3 numbBlocks(N/ threadsPerBlock.x,M/ threadsPerBlock.y); 
	

	sobel_abs_GPU<<<numbBlocks, threadsPerBlock>>>(rows,cols, gpudataSrc, gpudataOut, mode);
	cudaMemcpyAsync(dst, gpudataOut, size, cudaMemcpyDeviceToHost);
	
	cudaFree(gpudataSrc);
	cudaFree(gpudataOut);
}


void cv::gpu::mj::disp(const int rows,const int cols, unsigned char *g_srcL, unsigned char *g_srcR, unsigned char* g_disp, int shift){
	
	int N = WIDTH;
	int M = HEIGHT;
	
	//int* g_tmpL;
	//unsigned char* g_tmpOut;
	unsigned char* g_tmpL;
	unsigned char* g_tmpR;
	unsigned char* g_tmpL2;
	unsigned char* g_tmpR2;
	
	const int size = sizeof(unsigned char)*rows*cols;
	
	cudaMalloc((void **)&g_tmpL, size);
	cudaMalloc((void **)&g_tmpR, size);
	cudaMalloc((void **)&g_tmpL2, size);
	cudaMalloc((void **)&g_tmpR2, size);
	//cudaMalloc((void **)&g_tmpOut, size);
	//cudaMalloc((void **)&g_tmpOut, size);
	//cudaMalloc((void **)&g_tmp3L, size);
	cudaMemset(g_disp, 0, size);
	//cudaMemcpyAsync(g_srcR, srcR, size, cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(16, 16);
	dim3 numbBlocks(N/ threadsPerBlock.x,M/ threadsPerBlock.y); 
	//cudaDeviceSynchronize();
	//cout<<"1: Done!!!"<<endl;
	//ctoi_GPU<<<numbBlocks, threadsPerBlock>>>(rows , cols, g_srcL, g_tmpL);
  	//cudaDeviceSynchronize();
  	
	//prewittFS_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpL, g_tmpLX, g_tmpLY, g_tmp2LX, g_tmp2LY);
	
	edgeDetect_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_srcL, g_tmpL, 50);
	edgeDetect_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_srcR, g_tmpR, 50);
	//cudaDeviceSynchronize();
	edgeTypeDetect<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpL, g_tmpL2);
	edgeTypeDetect<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpR, g_tmpR2);
	cudaDeviceSynchronize();
	reduce<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpL2, g_tmpL);
	reduce<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpR2, g_tmpR);
	cudaDeviceSynchronize();
	edgeTypeDetect<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpL, g_tmpL2);
	edgeTypeDetect<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpR, g_tmpR2);
	cudaDeviceSynchronize();
	
	
	dim3 threadsPerBlockDisp(8, 8);
	dim3 numbBlocksDisp(N/ threadsPerBlockDisp.x,M/ threadsPerBlockDisp.y); 
	compare<<<numbBlocksDisp, threadsPerBlockDisp>>>(rows, cols, g_tmpL2, g_tmpR2, g_disp, shift);
	//blend_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpL, g_tmpR, g_disp, 0.5, 1);
	//findNode<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_disp, g_disp);
	
	//dim1 threadsPerBlock1(1);
	//dim1 numbBlocks1(1024/ threadsPerBlock.x); 
	
	//edgeDraw<<<200, 1>>>(rows, cols, g_tmpOut, g_disp);
	
  	//cudaDeviceSynchronize();
	//edgeDetect_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_srcR, g_srcR, 30);
  	/*prewittX_GPU <<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpL, g_tmpLX, 0);
  	prewittY_GPU <<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpL, g_tmpLY, 0);
  	prewittXsec_GPU <<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpL, g_tmp2LX, 0);
  	prewittYsec_GPU <<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpL, g_tmp2LY, 0);
  	cudaDeviceSynchronize();
  	*/
  	
	///edgeDetect<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_tmpLX, g_tmpLY, g_tmp2LX, g_tmp2LY, g_tmpOut);
	
	//itoc_GPU<<<numbBlocks, threadsPerBlock>>>(rows , cols, g_tmpOut, g_disp);
  	//cudaDeviceSynchronize();
	//cudaFree(g_tmpOut);
	cudaFree(g_tmpL);
	cudaFree(g_tmpR);
	cudaFree(g_tmpL2);
	cudaFree(g_tmpL2);
}


void cv::gpu::mj::rectif(const int rows,const int cols, unsigned char *srcL, unsigned char *srcR, unsigned char *dstL, unsigned char *dstR, unsigned char * out){
	
	
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

void cv::gpu::mj::cudaMemcpyHtoD(unsigned char *src, unsigned char* dest, int size){

	cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void cv::gpu::mj::cudaMemcpyDtoH(unsigned char *src, unsigned char* dest, int size){

	cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}

void cv::gpu::mj::cudaInit(unsigned char** g_src1, unsigned char** g_src2, unsigned char** g_disp, const int rows, const int cols){
	
	const int size = sizeof(unsigned char)*rows*cols;

	cudaMalloc((void **)g_src1, size);
	cudaMalloc((void **)g_src2, size);
	cudaMalloc((void **)g_disp, size);
	
}

void cv::gpu::mj::cudaDestroy(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp){
	
	cudaFree(g_src1);
	cudaFree(g_src2);
	cudaFree(g_disp);
}




























