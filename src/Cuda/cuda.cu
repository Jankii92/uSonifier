#include "cuda.h"
#include "improc.h"
#include "macher.h"
#include <stdio.h>
#include <chrono>
#include <device_functions.h>


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

unsigned char** cv::gpu::mj::initDisp(const int size){
	
	unsigned char** temps = (unsigned char**)malloc(8*sizeof(unsigned char*));
	
	int i = 0;
	
	for(i = 0; i < 8; i++){
		cudaMalloc((void **)&(temps[i]), size);
	} 
	return temps;
}
/*
void cv::gpu::mj::disp(const int rows,const int cols, unsigned char *g_srcL, unsigned char *g_srcR, unsigned char* g_disp, unsigned char** temps){
	
	int N = WIDTH;
	int M = HEIGHT;
	
	unsigned char* g_L_low_ext = temps[0];
	unsigned char* g_R_low_ext = temps[1];
	unsigned char* g_L_high_out = temps[2];
	unsigned char* g_R_high_out = temps[3];
	unsigned char* g_L_lowEdge = temps[4];
	unsigned char* g_R_lowEdge = temps[5];
	unsigned char* g_L_highEdge = temps[6];
	unsigned char* g_R_highEdge = temps[7];
	
	const int size = sizeof(unsigned char)*rows*cols;
	
	cudaMemset(g_disp, 0, size);
	
	dim3 threadsPerBlock(16, 16);
	dim3 numbBlocks(N/ threadsPerBlock.x,M/ threadsPerBlock.y); 
	
	edgeDetect2x_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_srcL, g_L_high_out, g_L_low_ext, 30, 3);
	edgeDetect2x_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_srcR, g_R_high_out, g_R_low_ext, 30, 3);
	
	cudaDeviceSynchronize();
	//cudaMemset(g_L_lowEdge, 0, size);
	//cudaMemset(g_L_low_ext, 0, size);
	//edgeTypeDetect<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_L_low_ext, g_L_lowEdge);
	edgeTypeDetectCleanup<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_L_high_out, g_L_lowEdge);
	edgeTypeDetectCleanup<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_R_high_out, g_R_lowEdge);
	cudaDeviceSynchronize();
	cudaMemset(g_L_high_out, 0, size);
	cudaMemset(g_R_high_out, 0, size);
	cudaDeviceSynchronize();
	edgeTypeDetectCleanup<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_L_lowEdge, g_L_high_out);
	edgeTypeDetectCleanup<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_R_lowEdge, g_R_high_out);
	//cudaMemset(g_L_highEdge, 0, size);
	//cudaMemset(g_R_highEdge, 0, size);
	cudaDeviceSynchronize();
	edgeTypeDetect<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_L_high_out, g_L_highEdge);
	edgeTypeDetect<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_R_high_out, g_R_highEdge);
	
	cudaDeviceSynchronize();
	
	cudaMemset(g_R_lowEdge, 0, size);
	cudaMemset(g_R_low_ext, 0, size);
	dim3 threadsPerBlockDisp(48, 16);
	dim3 numbBlocksDisp(N/16, M/16); 
	
	edgeMacher<<<numbBlocksDisp, threadsPerBlockDisp>>>( rows, cols, g_L_highEdge, g_R_highEdge, g_srcL, g_srcR, g_disp);
	
	cudaDeviceSynchronize();
	dim3 threadsPerBlock2(24, 24);
	dim3 numbBlocks2(N/ threadsPerBlock2.x,M/ threadsPerBlock2.y); 

	
	
	dim3 threadsPerBlockHori(WIDTH, 1);
	dim3 numbBlocksHori(N/WIDTH, M/1); 
	
	//filler<<<numbBlocksHori, threadsPerBlockHori>>>( rows, cols, g_R_low_ext, g_disp);
	//cudaDeviceSynchronize();
	//blend_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_L_lowEdge, g_R_lowEdge, g_disp, 0.5, 1);
	
	cudaMemset(g_L_low_ext, 0, size);
	cudaMemset(g_R_low_ext, 0, size);
	cudaMemset(g_L_high_out, 0, size);
	cudaMemset(g_R_high_out, 0, size);
	cudaMemset(g_L_lowEdge, 0, size);
	cudaMemset(g_R_lowEdge, 0, size);
	cudaMemset(g_L_highEdge, 0, size);
	cudaMemset(g_R_highEdge, 0, size);
	
}

unsigned char** cv::gpu::mj::initDisp2C(const int size){
	
	unsigned char** temps = (unsigned char**)malloc(3*sizeof(unsigned char*));

	for(int i = 0; i < 3; i++){
		cudaMalloc((void **)&(temps[i]), size);
	} 
	return temps;
}
__global__ void fillInitParams(unsigned int * t){
	if( threadIdx.x == 0 &&  threadIdx.y == 0){ 
		t[0 ] = 32; 
		t[1 ] = 32; 
		t[2 ] = 32; 
		t[3 ] = 31;
		t[4 ] = 30; 
		t[5 ] = 29; 
		t[6 ] = 28; 
		t[7 ] = 27;
		t[8 ] = 26; 
		t[9 ] = 25; 
		t[10] = 24; 
		t[11] = 23;
		t[12] = 22; 
		t[13] = 21; 
		t[14] = 20; 
		t[15] = 19;
		t[16] = 18; 
		t[17] = 17; 
		t[18] = 16; 
		t[19] = 17;
		t[20] = 16; 
		t[21] = 15; 
		t[22] = 16; 
		t[23] = 13; 
		t[24] = 12; 
	}

}
unsigned int** cv::gpu::mj::initDisp2I(const int size){
	
	unsigned int** temps = (unsigned int**)malloc(21*sizeof(unsigned int*));
	
	cudaMalloc((void **)&(temps[0]), size/(2*2)*64);
	cudaMalloc((void **)&(temps[1]), size/(4*4)*64);
	cudaMalloc((void **)&(temps[2]), size/(8*8)*64);
	cudaMalloc((void **)&(temps[3]), size/(16*16)*64);
	cudaMalloc((void **)&(temps[4]), size/(32*32)*64);
	cudaMalloc((void **)&(temps[5]), size/(2*2)*64);
	cudaMalloc((void **)&(temps[6]), size/(4*4)*64);
	cudaMalloc((void **)&(temps[7]), size/(8*8)*64);
	cudaMalloc((void **)&(temps[8]), size/(16*16)*64);
	cudaMalloc((void **)&(temps[9]), size/(32*32)*64);
	cudaMalloc((void **)&(temps[10]), size/(2*2)*64);
	cudaMalloc((void **)&(temps[11]), size/(4*4)*64);
	cudaMalloc((void **)&(temps[12]), size/(8*8)*64);
	cudaMalloc((void **)&(temps[13]), size/(16*16)*64);
	cudaMalloc((void **)&(temps[14]), size/(32*32)*64);
	cudaMalloc((void **)&(temps[15]), size/(2*2)*64);
	cudaMalloc((void **)&(temps[16]), size/(4*4)*64);
	cudaMalloc((void **)&(temps[17]), size/(8*8)*64);
	cudaMalloc((void **)&(temps[18]), size/(16*16)*64);
	cudaMalloc((void **)&(temps[19]), size/(32*32)*64);
	cudaMalloc((void **)&(temps[20]), 32*sizeof(unsigned int));
	cudaDeviceSynchronize();
	unsigned int * t = temps[20];
	
	fillInitParams<<<1, 1>>>(t);
	
	
	return temps;
}


void cv::gpu::mj::disp2(const int rows,const int cols, unsigned char *g_srcL, unsigned char *g_srcR, unsigned char* g_disp, unsigned char** tempsC, unsigned int** tempsI ){
	
	int N = WIDTH;
	int M = HEIGHT;
	
	unsigned char* g_L_edge = tempsC[0];
	unsigned char* g_R_edge = tempsC[1];
	unsigned char* g_edgeMached = tempsC[2];
	
	unsigned int* g_match_2 = tempsI[0];
	unsigned int* g_match_4 = tempsI[1];
	unsigned int* g_match_8 = tempsI[2];
	unsigned int* g_match_16 = tempsI[3];
	unsigned int* g_match_32 = tempsI[4];
	unsigned int* g_match_2x = tempsI[5];
	unsigned int* g_match_4x = tempsI[6];
	unsigned int* g_match_8x = tempsI[7];
	unsigned int* g_match_16x = tempsI[8];
	unsigned int* g_match_32x = tempsI[9];
	unsigned int* g_match_2y = tempsI[10];
	unsigned int* g_match_4y = tempsI[11];
	unsigned int* g_match_8y = tempsI[12];
	unsigned int* g_match_16y = tempsI[13];
	unsigned int* g_match_32y = tempsI[14];
	unsigned int* g_match_2xy = tempsI[15];
	unsigned int* g_match_4xy = tempsI[16];
	unsigned int* g_match_8xy = tempsI[17];
	unsigned int* g_match_16xy = tempsI[18];
	unsigned int* g_match_32xy = tempsI[19];
	unsigned int* g_params = tempsI[20];
	
	const int size = sizeof(unsigned char)*rows*cols;
	
	cudaMemset(g_disp, 0, size);
	cudaMemset(g_L_edge, 0, size);
	cudaMemset(g_R_edge, 0, size);

	cudaDeviceSynchronize();
	dim3 threadsPerBlock(16, 16);
	dim3 numbBlocks(N/ threadsPerBlock.x,M/ threadsPerBlock.y); 
	edgeDetect_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_srcL, g_L_edge, 50);
	//edgeDetect_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_srcR, g_R_edge, 40);
	
	cudaDeviceSynchronize();
	dim3 threadsPerBlock24(24, 24);
	dim3 numbBlocks24(N/ threadsPerBlock24.x,M/ threadsPerBlock24.y); 
	//findDistance<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_L_edge, g_R_edge);
	findDistanceFast<<<numbBlocks24, threadsPerBlock24,64*64*sizeof(unsigned char) >>>(rows, cols, g_L_edge, g_R_edge);
	cudaDeviceSynchronize();
	dim3 threadsPerBlockDisp2(64, 2);
	dim3 numbBlocksDisp2(N/2, M/2); 
	dim3 threadsPerBlockDisp4(64, 4);
	dim3 numbBlocksDisp4(N/4, M/4); 
	dim3 threadsPerBlockDisp8(64, 8);
	dim3 numbBlocksDisp8(N/8, M/8); 
	dim3 threadsPerBlockDisp12(64, 12);
	dim3 numbBlocksDisp12(N/12, M/12); 
	dim3 threadsPerBlockDisp16(64, 16);
	dim3 numbBlocksDisp16(N/16, M/16); 
	dim3 threadsPerBlockDisp32(64, 16);
	dim3 numbBlocksDisp32(N/32, M/32); 	
	
	edgeMatch2<<<numbBlocksDisp2, threadsPerBlockDisp2>>>( rows, cols, g_srcL, g_srcR, g_match_2, 0, 0);
	edgeMatch16<<<numbBlocksDisp16, threadsPerBlockDisp16, 64*16*sizeof(unsigned int)>>>( rows, cols, g_srcL, g_srcR, g_match_16, 0, 0);
	edgeMatch32<<<numbBlocksDisp32, threadsPerBlockDisp32, 64*32*sizeof(unsigned int)>>>( rows, cols, g_srcL, g_srcR, g_match_32, 0, 0);

	edgeMatch2<<<numbBlocksDisp2, threadsPerBlockDisp2>>>( rows, cols, g_srcL, g_srcR, g_match_2x, 1, 0);
	edgeMatch16<<<numbBlocksDisp16, threadsPerBlockDisp16, 64*16*sizeof(unsigned int)>>>( rows, cols, g_srcL, g_srcR, g_match_16x, 8, 0);
	edgeMatch32<<<numbBlocksDisp32, threadsPerBlockDisp32, 64*32*sizeof(unsigned int)>>>( rows, cols, g_srcL, g_srcR, g_match_32x, 16, 0);
	
	edgeMatch2<<<numbBlocksDisp2, threadsPerBlockDisp2>>>( rows, cols, g_srcL, g_srcR, g_match_2y, 0, 1);
	edgeMatch16<<<numbBlocksDisp16, threadsPerBlockDisp16, 64*16*sizeof(unsigned int)>>>( rows, cols, g_srcL, g_srcR, g_match_16y, 0, 8);
	edgeMatch32<<<numbBlocksDisp32, threadsPerBlockDisp32, 64*32*sizeof(unsigned int)>>>( rows, cols, g_srcL, g_srcR, g_match_32y, 0, 16);
	
	edgeMatch2<<<numbBlocksDisp2, threadsPerBlockDisp2>>>( rows, cols, g_srcL, g_srcR, g_match_2xy, 1, 1);
	edgeMatch16<<<numbBlocksDisp16, threadsPerBlockDisp16, 64*16*sizeof(unsigned int)>>>( rows, cols, g_srcL, g_srcR, g_match_16xy, 8, 8);
	edgeMatch32<<<numbBlocksDisp32, threadsPerBlockDisp32, 64*32*sizeof(unsigned int)>>>( rows, cols, g_srcL, g_srcR, g_match_32xy, 16, 16);
 	cudaDeviceSynchronize();
	
	dim3 threadsPerBlockDisp64(1, 1, 64);
	dim3 numbBlocksDisp64(N/1, M/1, 1); 
	brain1<<<numbBlocksDisp64, threadsPerBlockDisp64>>>( rows, cols, g_R_edge, g_match_2, g_match_4, g_match_8, g_match_16, g_match_32,  g_match_2x, g_match_4x, g_match_8x, g_match_16x, g_match_32x, g_match_2y, g_match_4y, g_match_8y, g_match_16y, g_match_32y, g_match_2xy, g_match_4xy, g_match_8xy, g_match_16xy, g_match_32xy, g_params, g_edgeMached);
	

	dim3 threadsPerBlock4(4, 4);
	dim3 numbBlocks4(N/ threadsPerBlock4.x,M/ threadsPerBlock4.y); 
	//median<<<numbBlocks4, threadsPerBlock4>>>(rows, cols, g_R_edge, g_L_edge, g_disp);
	
	cudaDeviceSynchronize();
	median5x5Edge<<<numbBlocks4, threadsPerBlock4>>>(rows, cols, g_L_edge, g_edgeMached,  g_disp);
	cudaDeviceSynchronize();
	
}*/

unsigned char** cv::gpu::mj::initDisp3C(const int size){
	
	unsigned char** temps = (unsigned char**)malloc(3*sizeof(unsigned char*));

	for(int i = 0; i < 3; i++){
		cudaMalloc((void **)&(temps[i]), size);
	} 
	return temps;
}
__global__ void fillInitParamsDisp3(unsigned short * t){
	if( threadIdx.x == 0 &&  threadIdx.y == 0){ 
	int multi = 3;
		t[0 ] = multi*45; 
		t[1 ] = multi*40; 
		t[2 ] = multi*35; 
		t[3 ] = multi*30;
		t[4 ] = multi*29; 
		t[5 ] = multi*28; 
		t[6 ] = multi*25; 
		t[7 ] = multi*20;
		t[8 ] = multi*18; 
		t[9 ] = multi*16; 
		t[10] = multi*14; 
		t[11] = multi*12;
		t[12] = multi*11; 
		t[13] = multi*10; 
		t[14] = multi*9; 
		t[15] = multi*8;
		t[16] = multi*7; 
		t[17] = multi*6; 
		t[18] = multi*5; 
		t[19] = multi*4;
		t[20] = multi*3; 
		t[21] = multi*2; 
		t[22] = multi*1; 
		t[23] = multi*1; 
		t[24] = multi*1; 
	}

}
unsigned short** cv::gpu::mj::initDisp3US(const int size){
	
	unsigned short** temps = (unsigned short**)malloc(2*sizeof(unsigned short*));
	
	cudaMalloc((void **)&(temps[0]), size/(8*8)*64);
	cudaMalloc((void **)&(temps[1]), 32*sizeof(unsigned short));
	cudaDeviceSynchronize();
	unsigned short * t = temps[1];
	
	fillInitParamsDisp3<<<1, 1>>>(t);
	
	return temps;
}


void cv::gpu::mj::disp3(const int rows,const int cols, unsigned char *g_srcL, unsigned char *g_srcR, unsigned char* g_disp, unsigned char** tempsC, unsigned short** tempsUS ){
	
	int w = WIDTH;
	int h = HEIGHT;
	
	size_t offset = 0;
	
	unsigned char* g_L_edge = tempsC[0];
	unsigned char* g_R_edge = tempsC[1];
	//unsigned char* g_udisp = tempsC[2];
	
	unsigned short* g_match_8 = tempsUS[0];
	unsigned short* g_w = tempsUS[1];
	
	const int size = sizeof(unsigned char)*rows*cols;
	
	cudaMemset(g_disp, 0, size);
	cudaMemset(g_L_edge, 0, size);
	cudaMemset(g_R_edge, 0, size);
    cudaBindTexture2D(&offset, tex2Dleft,  g_srcL, ca_desc0, w, h, w*4);
    cudaBindTexture2D(&offset, tex2Dright, g_srcR, ca_desc1, w, h, w*4);
    
	cudaDeviceSynchronize();
	dim3 threadsPerBlock(16, 16);
	dim3 numbBlocks(w/ threadsPerBlock.x,h/ threadsPerBlock.y); 
	edgeDetect_GPU<<<numbBlocks, threadsPerBlock>>>(rows, cols, g_srcL, g_L_edge, 50);
	cudaDeviceSynchronize();
	dim3 threadsPerBlock24(24, 24);
	dim3 numbBlocks24(w/ threadsPerBlock24.x,h/ threadsPerBlock24.y); 
	findDistanceFast<<<numbBlocks24, threadsPerBlock24,64*64*sizeof(unsigned char) >>>(rows, cols, g_L_edge, g_R_edge);
	cudaDeviceSynchronize();
	dim3 threadsPerBlockDisp16(64, 16);
	dim3 numbBlocksDisp16(w/16, h/16); 
	edgeMatch8w16<<<numbBlocksDisp16,threadsPerBlockDisp16,16*64*2*sizeof(unsigned short)>>>(rows, cols, g_srcL, g_srcR, g_match_8);	
 	dim3 threadsPerBlockDisp64(16, 16, 2);
	dim3 numbBlocksDisp64(w/16, h/16, 1); 
 	int extSize = 3*3*64*sizeof(unsigned int)+6*6*64*sizeof(unsigned short)+16*16*64*sizeof(unsigned short)+4*4*64*sizeof(unsigned int)+4*4*32*sizeof(unsigned char)+4*4*32*sizeof(unsigned int);
 	brain3<<<numbBlocksDisp64, threadsPerBlockDisp64, extSize>>>( rows, cols, g_srcL, g_srcR, g_R_edge, g_match_8, g_w, g_disp, 1, 1000*256);

 	cudaUnbindTexture(tex2Dleft);
 	cudaUnbindTexture(tex2Dright);
 	//perform Extended 2x2 maching 
 	//
	//	1|2|1
	//	2|4|2
	//	1|2|1
	//
	// and mix it with proportional 16 and 32 block 
	
}
void cv::gpu::mj::dispToUdepth(const int rows,const int cols, const int uRows,const int uCols, unsigned char *g_disp, unsigned char *g_udepth, unsigned char** tempsC){
	
	int N = WIDTH;
	int M = HEIGHT;
	
	unsigned char* g_udisp = tempsC[0];
	const int size = sizeof(unsigned char)*rows*cols;
	cudaMemset(g_udisp, 0, size);
	 	
	dim3 threadsPerBlockUDisp(1, 480);
	dim3 numbBlocksUDisp(N/1, M/480); 
	udisp<<<threadsPerBlockUDisp, numbBlocksUDisp>>>(rows, cols, g_disp, g_udisp);
	cudaDeviceSynchronize();
	cudaMemset(g_udepth, 0, size);
	dim3 threadsPerBlockUDepth(1, 256);
	dim3 numbBlocksUDepth(N/1, M/256);
	udispToUdepth<<<threadsPerBlockUDepth,numbBlocksUDepth>>>(uRows, uCols, g_udisp, g_udepth);
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
	
	size_t offset = 0;
    
	const int size = sizeof(unsigned char)*rows*cols;

	cudaMalloc((void **)g_src1, size);
	cudaMalloc((void **)g_src2, size);
	cudaMalloc((void **)g_disp, size);
	
	tex2Dleft.addressMode[0] = cudaAddressModeClamp;
    tex2Dleft.addressMode[1] = cudaAddressModeClamp;
    tex2Dleft.filterMode     = cudaFilterModePoint;
    tex2Dleft.normalized     = false;
    tex2Dright.addressMode[0] = cudaAddressModeClamp;
    tex2Dright.addressMode[1] = cudaAddressModeClamp;
    tex2Dright.filterMode     = cudaFilterModePoint;
    tex2Dright.normalized     = false;

	
}

void cv::gpu::mj::cudaDestroy(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp, unsigned char** g_temps){
	
	cudaFree(g_src1);
	cudaFree(g_src2);
	cudaFree(g_disp);
	int i;
	for(i = 0; i < 8; i++){
		cudaFree(g_temps[i]);
	} 
}

void cv::gpu::mj::cudaDestroyDisp2(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp, unsigned char** g_tempsC, unsigned int** g_tempsI){
	
	cudaFree(g_src1);
	cudaFree(g_src2);
	cudaFree(g_disp);
	int i;
	for(i = 0; i < 3; i++){
		cudaFree(g_tempsC[i]);
	} 
	for(i = 0; i < 20; i++){
		cudaFree(g_tempsI[i]);
	} 
}

void cv::gpu::mj::cudaDestroyDisp3(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp, unsigned char** g_tempsC, unsigned short** g_tempsUC){
	
	cudaFree(g_src1);
	cudaFree(g_src2);
	cudaFree(g_disp);
	int i;
	for(i = 0; i < 3; i++){
		cudaFree(g_tempsC[i]);
	} 
	for(i = 0; i < 2; i++){
		cudaFree(g_tempsUC[i]);
	} 
}




























