#include "improc.h"
#include <math.h>


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


__global__ void rectify_GPU (const int rows, const int cols, const unsigned char *imgL,const unsigned char *imgR, int *rot){
		//int x = blockIdx.x * blockDim.x + threadIdx.x;
    	//int y = blockIdx.y * blockDim.y + threadIdx.y;
    	
}

__global__ void sobel_GPU (const int rows, const int cols, const unsigned char *img, unsigned char *des, const int mode){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;
    	
    	if(x <= 1 || x >= cols-1 || y <= 1 || y >= rows-1) return;
    	int sum = 0;
    	if(mode == 0) {
    		sum-=img[(y-1)*cols+(x-1)];
    		//sum+=((int)img[(y-1)*cols+(x)]*sX[1];
    		sum+=img[(y-1)*cols+(x+1)];
    		sum-=(int)img[(y)*cols+(x-1)]*2;
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[4];
    		sum+=(int)img[(y)*cols+(x+1)]*2;
    		sum-=img[(y+1)*cols+(x-1)];
    		//sum+=((int)img[(y+1)*cols+(x)]*sX[7];
    		sum+=img[(y+1)*cols+(x+1)];
    		des[y*cols+x] = (unsigned char)(sum/9+128);
    	}else if(mode == 1){
    		sum+=img[(y-1)*cols+(x-1)];
    		sum+=(int)img[(y-1)*cols+(x)]*2;
    		sum+=img[(y-1)*cols+(x+1)];
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[3];
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[4];
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[5];
    		sum-=img[(y+1)*cols+(x-1)];
    		sum-=(int)img[(y+1)*cols+(x)]*2;
    		sum-=img[(y+1)*cols+(x+1)];
    		des[y*cols+x] = (unsigned char)(sum/9+128);
    	}    	
}

__global__ void sobel_abs_GPU (const int rows, const int cols, const unsigned char *img, unsigned char *des, const int mode){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;
    	
    	if(x <= 1 || x >= cols-1 || y <= 1 || y >= rows-1) return;
    	int sum = 0;
    	if(mode == 0) {
    		sum-=img[(y-1)*cols+(x-1)];
    		//sum+=((int)img[(y-1)*cols+(x)]*sX[1];
    		sum+=img[(y-1)*cols+(x+1)];
    		sum-=(int)img[(y)*cols+(x-1)]*2;
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[4];
    		sum+=(int)img[(y)*cols+(x+1)]*2;
    		sum-=img[(y+1)*cols+(x-1)];
    		//sum+=((int)img[(y+1)*cols+(x)]*sX[7];
    		sum+=img[(y+1)*cols+(x+1)];
    		if(sum < 0) des[y*cols+x] = 2*(unsigned char)(-sum/9);
    		else des[y*cols+x] = 2*(unsigned char)(sum/9);
    	}else if(mode == 1){
    		sum+=img[(y-1)*cols+(x-1)];
    		sum+=(int)img[(y-1)*cols+(x)]*2;
    		sum+=img[(y-1)*cols+(x+1)];
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[3];
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[4];
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[5];
    		sum-=img[(y+1)*cols+(x-1)];
    		sum-=(int)img[(y+1)*cols+(x)]*2;
    		sum-=img[(y+1)*cols+(x+1)];
    		if(sum < 0) des[y*cols+x] = (unsigned char)(-sum/9);
    		else des[y*cols+x] = (unsigned char)(sum/9);
    	}    	
}

__global__ void rotate_GPU (const int rows, const int cols, const unsigned char *img, unsigned char *des, float deg){
		//int x = blockIdx.x * blockDim.x + threadIdx.x;
    	//int y = blockIdx.y * blockDim.y + threadIdx.y;
    	
    	//if(x < 0  || x > cols || y <= 1 || y >= rows-1) return;
    	
    	int i = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
    	int j = blockIdx.y * blockDim.y + threadIdx.y;
    	int xc = cols - cols/2;
    	int yc = rows - rows/2;
    	int newx = ((float)i-xc)*cos(deg) - ((float)j-yc)*sin(deg) + xc;
    	int newy = ((float)i-xc)*sin(deg) + ((float)j-yc)*cos(deg) + yc;
		if (newx >= 0 && newx < cols && newy >= 0 && newy < rows)
		{
		    des[j*cols+i] = img[newy*cols+newx];
		}
}


__global__ void blend_GPU (const int rows, const int cols, const unsigned char *img1, const unsigned char *img2, unsigned char *dest, const float blend){
	int x = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x < 0  || x > cols || y < 0 || y > rows) return;
	
	dest[y*cols+x] = (unsigned char)(blend*img1[y*cols+x])+(unsigned char)((1-blend)*img2[y*cols+x]); 
}


