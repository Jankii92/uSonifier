#include "improc.h"
#include <math.h>
#include <stdlib.h>
#include <curand.h>
#include <iostream>

#define BLUR_K 5

#define EEND  50
#define END_X  150
#define END_Y  151
#define END_XY  152
#define END_YX  153
#define END_X_XY  154
#define END_X_YX  155
#define END_Y_XY  156
#define END_Y_YX  157
#define VERT  50
#define HORI  70
#define DIAG  90
#define NODE  255
#define NOISE  1


__device__ void share_uchar(int* in, int* out, const int in_rows, const int in_cols, const int blockDimm, int th_x, int th_y, int x, int y, const int r){
	
    	int loc_X = th_x+r;
    	int loc_Y = th_y+r; 
    	const int loc_Dim = blockDimm + 2*r;
 			
    	out[loc_Y * loc_Dim + loc_X] = in[y * in_cols + x];
    	
    	if(th_x < r) {
    		out[loc_Y * loc_Dim + loc_X - r] = in[y * in_cols + x - r];
    		if(th_y < r){
				out[(loc_Y-r) * loc_Dim + loc_X - r] = in[(y-r)*in_cols+x-r];
    		}
		}
		if(th_x > blockDimm - r-1){
    		out[loc_Y * loc_Dim + loc_X + r] = in[y * in_cols + x + r];
    		if(th_y > blockDimm - r - 1){
    			out[(loc_Y+r) * loc_Dim + loc_X + r] = in[(y+r)*in_cols+x+r];
    		}
		}
    	if(th_y < r) {
    		out[(loc_Y-r) * loc_Dim + loc_X] = in[(y-r) * in_cols + x];
    		if(th_x > blockDimm - r-1){
    			out[(loc_Y-r) * loc_Dim + loc_X + r] = in[(y-r)*in_cols+x+r];
    		}
		}
		if(th_y > blockDimm - r-1){
    		out[(loc_Y+r) * loc_Dim + loc_X] = in[(y+r) * in_cols + x];
    		if(th_x < r){
    			out[(loc_Y+r) * loc_Dim + loc_X - r] = in[(y+r)*in_cols+x-r];
    		}
		}
    	__syncthreads();
}

__device__ bool SameSign(int x, int y)
{
    return ((x > 0) ^ (y < 0));
}

__device__ bool SameSign_GPU(int *x, int *y)
{
    return ((*x > 0) ^ (*y < 0));
}

__device__ void shift_GPU(char *x)
{
	*x =(*x << 1 | *x >> 7); 
}

__device__ bool isDeadEnd(unsigned char *x){
	if(*x == 7 || *x == 28 || *x == 112 || *x == 192)
		return true;
	else
		return false;
}

__device__ bool isEndEx(unsigned char *x){
	
	if(!((   (*x ^ 0b10000000) & 0b10111111) && 
			((*x ^ 0b01000000) & 0b01111111) &&
			((*x ^ 0b00100000) & 0b10111111) &&
			((*x ^ 0b00010000) & 0b11011111) &&
			((*x ^ 0b00001000) & 0b11101111) &&
			((*x ^ 0b00000100) & 0b11110111) &&
			((*x ^ 0b00000010) & 0b11111011) &&
			((*x ^ 0b00000001) & 0b11111101) &&
			((*x ^ 0b10000001) & 0b11111101)) || *x == 60 || *x == 104)
		return true;
	else 
		return false;
}
__device__ bool isEndX(unsigned char *x){
	
	if(*x == 8 || *x == 128) return true;
	else return false;
}
__device__ bool isEndY(unsigned char *x){
	
	if(*x == 2 || *x == 32) return true;
	else return false;
}
__device__ bool isEndXY(unsigned char *x){
	if(*x == 1 || *x == 16) return true;
	else return false;
}
__device__ bool isEndYX(unsigned char *x){
	if(*x == 4 || *x == 64) return true;
	else return false;
}
__device__ bool isEndXXY(unsigned char *x){
	if(*x == 24 || *x == 129) return true;
	else return false;
}
__device__ bool isEndXYX(unsigned char *x){
	if(*x == 12 || *x == 129) return true;
	else return false;
}
__device__ bool isEndYXY(unsigned char *x){
	
	if(*x == 4 || *x == 48) return true;
	else return false;
}
__device__ bool isEndYYX(unsigned char *x){
	
	if(*x == 6 || *x == 96) return true;
	else return false;
}



__device__ bool isVertical(unsigned char *x){
			//    ABCDEFGH      ABCDEFGH
	if(!(((*x ^ 0b00010010) & 0b11011010) &&
		 ((*x ^ 0b00010010) & 0b11010110) &&
		 ((*x ^ 0b00100001) & 0b01101101) &&
		 ((*x ^ 0b00100010) & 0b11101010) &&			
			//	  ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b00100100) & 0b10110101) &&	
		 ((*x ^ 0b01000010) & 0b11011010) &&
		 ((*x ^ 0b01000010) & 0b01011110) &&
		 ((*x ^ 0b01000010) & 0b01011011) &&		
			//	  ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b01100010) & 0b01111010) &&	
		 ((*x ^ 0b00010001) & 0b11011101) &&
		 ((*x ^ 0b00010011) & 0b01011111) &&
		 ((*x ^ 0b00010100) & 0b11011101) &&	
			//	  ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b00100110) & 0b10101111) &&	
		 ((*x ^ 0b00110001) & 0b11110101) &&
		 ((*x ^ 0b00110011) & 0b01110111) &&
		 ((*x ^ 0b01000001) & 0b11011101) &&
			//	  ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b01000100) & 0b11011101) &&	
		 ((*x ^ 0b01000110) & 0b11010111) &&
		 ((*x ^ 0b01100100) & 0b01111101) &&
		 ((*x ^ 0b01100110) & 0b01101111) &&
			//	  ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b10100011) & 0b10101111) &&	
		 ((*x ^ 0b11111000) & 0b11111110) &&
		 ((*x ^ 0b11111000) & 0b11111101)))
		return true;
	else 
		return false;
}

__device__ bool isHorizontal(unsigned char *x){
			//    ABCDEFGH      ABCDEFGH
	if(!(((*x ^ 0b10001000) & 0b10101010) &&
		 ((*x ^ 0b00001001) & 0b01101101) &&
		 ((*x ^ 0b00001001) & 0b01101011) &&
		 ((*x ^ 0b01001000) & 0b01101011) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b10000100) & 0b10110110) &&
		 ((*x ^ 0b10000100) & 0b10110101) &&
		 ((*x ^ 0b10010000) & 0b11010110) &&
		 ((*x ^ 0b10010000) & 0b10110110) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b10010000) & 0b10010111) &&
		 ((*x ^ 0b10011000) & 0b10011110) &&
		 ((*x ^ 0b11001000) & 0b11011011) &&
		 ((*x ^ 0b00000101) & 0b01110111) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b00010001) & 0b01110111) &&
		 ((*x ^ 0b00011001) & 0b01011111) &&
		 ((*x ^ 0b01000100) & 0b01110111) &&
		 ((*x ^ 0b01001100) & 0b01011111) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b01010000) & 0b01110111) &&
		 ((*x ^ 0b10010001) & 0b11110101) &&
		 ((*x ^ 0b10011001) & 0b11111001) &&
		 ((*x ^ 0b10011001) & 0b11011101) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b11000100) & 0b11010111) &&
		 ((*x ^ 0b11001100) & 0b11111100) &&
		 ((*x ^ 0b11001100) & 0b11101101) &&
		 ((*x ^ 0b11001100) & 0b11011101) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b01110001) & 0b11111101) &&
		 ((*x ^ 0b01110010) & 0b01111111) &&
		 ((*x ^ 0b01110011) & 0b11110111) &&
		 ((*x ^ 0b01110100) & 0b11111111)))		 
		return true;
	else 
		return false;
}

__device__ bool isNode3(unsigned char *x){
			//    ABCDEFGH      ABCDEFGH
	if(!(((*x ^ 0b01111101) & 0b10100111) &&
		 ((*x ^ 0b11101101) & 0b00110111) &&
		 ((*x ^ 0b01101111) & 0b10111001) &&
		 ((*x ^ 0b11101011) & 0b00111101) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b01011111) & 0b11101001) &&
		 ((*x ^ 0b01111011) & 0b11001101) &&
		 ((*x ^ 0b11011011) & 0b01101110) &&
		 ((*x ^ 0b11111010) & 0b01001111) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b01011111) & 0b11110010) &&
		 ((*x ^ 0b11010111) & 0b01111010) &&
		 ((*x ^ 0b10111110) & 0b11010011) &&
		 ((*x ^ 0b11110110) & 0b10011011) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b10110111) & 0b11011100) &&
		 ((*x ^ 0b11110101) & 0b10011110) &&
		 ((*x ^ 0b10101111) & 0b11110100) &&
		 ((*x ^ 0b10111101) & 0b11100110) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b10110101) & 0b01011111) &&
		 ((*x ^ 0b01101011) & 0b10111110) &&
		 ((*x ^ 0b01101101) & 0b11010111) &&
		 ((*x ^ 0b01011011) & 0b11110101) &&	
			//    ABCDEFGH      ABCDEFGH
		 ((*x ^ 0b11010110) & 0b01111101) &&
		 ((*x ^ 0b11011010) & 0b10101111) &&
		 ((*x ^ 0b10110110) & 0b11101011))) 
		return true;
	else 
		return false;
}


__global__ void ctoi_GPU(const int rows,const  int cols, unsigned char *img, int *result){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    result[y*cols+x] = img[y*cols+x];
}

__global__ void itoc_GPU(const int rows,const  int cols, int *img, unsigned char *result){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int value = img[y*cols+x];
    if(value > 255)
    	result[y*cols+x] = 255;
    else if(value < 0)
    	result[y*cols+x] = 0;
    else
    	result[y*cols+x] = (unsigned char)value;
}


__global__ void blur_GPU(const int rows,const  int cols, const int k, int *img, int *result){

    
	__shared__ int s[(16+BLUR_K-1)*(16+BLUR_K-1)];
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
   	int r = BLUR_K/2;
    
	if(x < r || x > cols-r || y < r  || y > rows-r ) return ;
    
    share_uchar(img, s, rows, cols, blockDim.x, threadIdx.x, threadIdx.y, x, y, BLUR_K/2);
    
    if(x < BLUR_K-1 || x > cols-BLUR_K || y < BLUR_K  || y > rows-BLUR_K ) return;
    
	int sum = 0;
	int xx;
	int yy;
	int rr = BLUR_K*BLUR_K;
	int th_x = threadIdx.x+r;
	int th_y = threadIdx.y+r;
	int th_dim = blockDim.x+2*r;
	
    xx = th_x-r;
	do{
		yy = th_y-r;
		do{
			sum+=s[yy*th_dim+xx];
		}while(yy++ < th_y+r);
	}while(xx++ < th_x+r);
	result[y*cols+x] = (int)round((float)sum/rr);	
}

__global__ void sobel_GPU (const int rows, const int cols, unsigned char *img, unsigned char *des, const int mode){
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
    	}   
    	des[y*cols+x] = (unsigned char)(sum/9+128); 	
}

__global__ void sobel_abs_GPU (const int rows, const int cols, unsigned char *img, unsigned char *des, const int mode){
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
    		
    	}else if(mode == 1){
    		sum-=img[(y-1)*cols+(x-1)];
    		sum-=(int)img[(y-1)*cols+(x)]*2;
    		sum-=img[(y-1)*cols+(x+1)];
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[3];
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[4];
    		//sum+=((int)img[(y)*cols+(x-1)]*sX[5];
    		sum+=img[(y+1)*cols+(x-1)];
    		sum+=(int)img[(y+1)*cols+(x)]*2;
    		sum+=img[(y+1)*cols+(x+1)];
    	}  
    	if(sum < 0) des[y*cols+x] = (unsigned char)(-sum/9.0);
    	else des[y*cols+x] = (unsigned char)(sum/9.0);	
}


__global__ void prewitt_GPU (const int rows, const int cols, int *img, int *des, const int mode){
		
		//const int r = 1;
		
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x <= 1 || x >= cols-1 || y <= 1 || y >= rows-1) return;
		
    	
    	int sumX = 0;
    	int sumY = 0;
    	    	
    	sumX-=img[(y-1)*cols+(x-1)];
    	//sum+=((int)img[(y-1)*cols+(x)]*sX[1];
    	sumX+=img[(y-1)*cols+(x+1)];
    	sumX-=img[(y)*cols+(x-1)];
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[4];
    	sumX+=img[(y)*cols+(x+1)]; 
    	sumX-=img[(y+1)*cols+(x-1)];
    	//sum+=((int)img[(y+1)*cols+(x)]*sX[7];
    	sumX+=img[(y+1)*cols+(x+1)];
    	

    	sumY-=img[(y-1)*cols+(x-1)];
    	sumY-=img[(y-1)*cols+(x)];
    	sumY-=img[(y-1)*cols+(x+1)];
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[3];
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[4];
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[5];
    	sumY+=img[(y+1)*cols+(x-1)];
    	sumY+=img[(y+1)*cols+(x)];
    	sumY+=img[(y+1)*cols+(x+1)];

		if(mode == 0) 
			des[y*cols+x] = (sumY+sumX);
		if(mode == 2)
			des[y*cols+x] = (abs(sumX)+abs(sumY));	
}


__global__ void prewittX_GPU (const int rows, const int cols, int *img, int *des, const int mode){
		
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x <= 1 || x >= cols-1 || y <= 1 || y >= rows-1) return;

    	int sumX = 0;
    	    	
    	sumX-=img[(y-1)*cols+(x-1)];
    	sumX+=img[(y-1)*cols+(x+1)];
    	sumX-=2*img[(y)*cols+(x-1)];
    	sumX+=2*img[(y)*cols+(x+1)]; 
    	sumX-=img[(y+1)*cols+(x-1)];
    	sumX+=img[(y+1)*cols+(x+1)];
    	
		if(mode == 0) 
			des[y*cols+x] = (sumX);
		else if(mode == 2)
			des[y*cols+x] = abs(sumX);	

}

__global__ void prewittXsec_GPU (const int rows, const int cols, int *img, int *des, const int mode){
		
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x <= 2 || x >= cols-2 || y <= 2 || y >= rows-2) return;

    	int sumX = 0;    	
    	
    	sumX+=img[(y-2)*cols+(x-2)];
    	sumX-=2*img[(y-2)*cols+(x)];
    	sumX+=img[(y-2)*cols+(x+2)];
    	
    	sumX+=4*img[(y-1)*cols+(x-2)];
    	sumX-=8*img[(y-1)*cols+(x)];
    	sumX+=4*img[(y-1)*cols+(x+2)];
    	
    	sumX+=6*img[(y)*cols+(x-2)];
    	sumX-=12*img[(y)*cols+(x)];
    	sumX+=6*img[(y)*cols+(x+2)];
    	
    	sumX+=4*img[(y+1)*cols+(x-2)];
    	sumX-=8*img[(y+1)*cols+(x)];
    	sumX+=4*img[(y+1)*cols+(x+2)];
    	
    	sumX+=img[(y+2)*cols+(x-2)];
    	sumX-=2*img[(y+2)*cols+(x)];
    	sumX+=img[(y+2)*cols+(x+2)];
    	
    	
		if(mode == 0) 
			des[y*cols+x] = (sumX);
		else if(mode == 2)
			des[y*cols+x] = abs(sumX);	

}

__global__ void prewittY_GPU (const int rows, const int cols, int *img, int *des, const int mode){
		
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x <= 1 || x >= cols-1 || y <= 1 || y >= rows-1) return;
    	
    	int sumY = 0;
    	
    	sumY-=img[(y-1)*cols+(x-1)];
    	sumY-=2*img[(y-1)*cols+(x)];
    	sumY-=img[(y-1)*cols+(x+1)];
    	sumY+=img[(y+1)*cols+(x-1)];
    	sumY+=2*img[(y+1)*cols+(x)];
    	sumY+=img[(y+1)*cols+(x+1)];

		if(mode == 0) 
			des[y*cols+x] = sumY;
		if(mode == 2)
			des[y*cols+x] = abs(sumY);	
}

__global__ void prewittFS_GPU (const int rows, const int cols, int *img, int *desXf, int *desYf, int *desXs, int *desYs){

    	__shared__ int s[20][20];
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+2;
		int iY = threadIdx.y+2;
    	
    	if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 4){
    		if(x < 2) s[iX-2][iY] = 0;
    		else{
    			s[iX-2][iY] = img[y*cols+x-2];
    			if(y < 2) s[iX-2][iY-2] = 0;
    			else s[iX-2][iY-2] = img[(y-2)*cols+(x-2)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-2) s[iX+2][iY] = 0;
    		else{
    			s[iX+2][iY] = img[y*cols+(x+2)];
    			if(y >= rows-2) s[iX+2][iY+2] = 0;
    			else s[iX+2][iY+2] = img[(y+2)*cols+x+2];
    		}
    	}
    	if(iY < 4){
    		if(y < 2) s[iX][iY-2] = 0;
    		else{
    			s[iX][iY-2] = img[(y-2)*cols+x];
    			if(x >= cols-2) s[iX+2][iY-2] = 0;
    			else s[iX+2][iY-2] = img[(y-2)*cols+x+2];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-2) s[iX][iY+2] = 0;
    		else{
    			s[iX][iY+2] = img[(y+2)*cols+x];
    			if(x < 2) s[iX-2][iY+2] = 0;
    			else s[iX-2][iY+2] = img[(y+2)*cols+(x-2)];
    		}
    	}
    	
    	__syncthreads();
    		
    	
    	int sumYf = 0;
    	int sumYs = 0;    	
    	int sumXf = 0;
    	int sumXs = 0;
		
		
    	sumYf-=  s[iX-1][iY-1];
    	sumYf-=2*s[iX  ][iY-1];
    	sumYf-=  s[iX+1][iY-1];
    	sumYf+=  s[iX-1][iY+1];
    	sumYf+=2*s[iX  ][iY+1];
    	sumYf+=  s[iX+1][iY+1];
    	
    	desYf[y*cols+x] = sumYf;
    	
    	sumYs+=  s[iX-2][iY-2];
    	sumYs+=4*s[iX-1][iY-2];
    	sumYs+=6*s[iX  ][iY-2];
    	sumYs+=4*s[iX+1][iY-2];
    	sumYs+=  s[iX+2][iY-2];

		sumYs-= 2*s[iX-2][iY];
    	sumYs-= 8*s[iX-1][iY];
    	sumYs-=12*s[iX  ][iY];
    	sumYs-= 8*s[iX+1][iY];
    	sumYs-= 2*s[iX+2][iY];
    	
    	sumYs+=  s[iX-2][iY+2];
    	sumYs+=4*s[iX-1][iY+2];
    	sumYs+=6*s[iX  ][iY+2];
    	sumYs+=4*s[iX+1][iY+2];
    	sumYs+=  s[iX+2][iY+2];

		desYs[y*cols+x] = sumYs;    	
    	
		sumXf-=  s[iX-1][iY-1];
    	sumXf-=2*s[iX-1][iY  ];
    	sumXf-=  s[iX-1][iY+1];
    	sumXf+=  s[iX+1][iY-1];
    	sumXf+=2*s[iX+1][iY  ];
    	sumXf+=  s[iX+1][iY+1];
    	
    	desXf[y*cols+x] = sumXf;
    	
    	sumXs+=  s[iX-2][iY-2];
    	sumXs+=4*s[iX-2][iY-1];
    	sumXs+=6*s[iX-2][iY  ];
    	sumXs+=4*s[iX-2][iY+1];
    	sumXs+=  s[iX-2][iY+2];

		sumXs-= 2*s[iX][iY-2];
    	sumXs-= 8*s[iX][iY-1];
    	sumXs-=12*s[iX][iY  ];
    	sumXs-= 8*s[iX][iY+1];
    	sumXs-= 2*s[iX][iY+2];
    	
    	sumXs+=  s[iX+2][iY-2];
    	sumXs+=4*s[iX+2][iY-1];
    	sumXs+=6*s[iX+2][iY  ];
    	sumXs+=4*s[iX+2][iY+1];
    	sumXs+=  s[iX+2][iY+2];

		desXs[y*cols+x] = sumXs; 
		
}

__global__ void edgeDetect_GPU (const int rows, const int cols, unsigned char *img, unsigned char *out, int th){

    	__shared__ unsigned char s[22][22];
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+3;
		int iY = threadIdx.y+3;
    	
    	if(x < 0 || x >= cols-1 || y < 0 || y >= rows-1) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 6){
    		if(x < 3) s[iX-3][iY] = 0;
    		else{
    			s[iX-3][iY] = img[y*cols+x-3];
    			if(y < 3) s[iX-3][iY-3] = 0;
    			else s[iX-3][iY-3] = img[(y-3)*cols+(x-3)];  		
    		}
    	}
    	if(iX >= blockDim.x-1){
    		if(x >= cols-3) s[iX+3][iY] = 0;
    		else{
    			s[iX+3][iY] = img[y*cols+(x+3)];
    			if(y >= rows-3) s[iX+3][iY+3] = 0;
    			else s[iX+3][iY+3] = img[(y+3)*cols+x+3];
    		}
    	}
    	if(iY < 6){
    		if(y < 3) s[iX][iY-3] = 0;
    		else{
    			s[iX][iY-3] = img[(y-3)*cols+x];
    			if(x >= cols-3) s[iX+3][iY-3] = 0;
    			else s[iX+3][iY-3] = img[(y-3)*cols+x+3];
    		}
    	}
    	if(iY >= blockDim.y-1){
    		if(y >= rows-3) s[iX][iY+3] = 0;
    		else{
    			s[iX][iY+3] = img[(y+3)*cols+x];
    			if(x < 3) s[iX-3][iY+3] = 0;
    			else s[iX-3][iY+3] = img[(y+3)*cols+(x-3)];
    		}
    	}
    	
    	__syncthreads();
    		
    	//if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	int sumYf = 0;
    	int sumYs = 0; 
    	int sumYss = 0;   
    	int sumYsm = 0;   	
    	int sumXf = 0;
    	int sumXs = 0;
    	int sumXss = 0;
    	int sumXsm = 0;
  
// Y first		
    	sumYf-=  s[iX-1][iY-1];
    	sumYf-=2*s[iX  ][iY-1];
    	sumYf-=  s[iX+1][iY-1];
    	sumYf+=  s[iX-1][iY+1];
    	sumYf+=2*s[iX  ][iY+1];
    	sumYf+=  s[iX+1][iY+1];
// Y second    	
    	sumYs+=  s[iX-2][iY-2];
    	sumYs+=4*s[iX-1][iY-2];
    	sumYs+=6*s[iX  ][iY-2];
    	sumYs+=4*s[iX+1][iY-2];
    	sumYs+=  s[iX+2][iY-2];

		sumYs-= 2*s[iX-2][iY];
    	sumYs-= 8*s[iX-1][iY];
    	sumYs-=12*s[iX  ][iY];
    	sumYs-= 8*s[iX+1][iY];
    	sumYs-= 2*s[iX+2][iY];
    	
    	sumYs+=  s[iX-2][iY+2];
    	sumYs+=4*s[iX-1][iY+2];
    	sumYs+=6*s[iX  ][iY+2];
    	sumYs+=4*s[iX+1][iY+2];
    	sumYs+=  s[iX+2][iY+2];
// Y-1 second  	
		sumYsm+=  s[iX-2][iY-3];
    	sumYsm+=4*s[iX-1][iY-3];
    	sumYsm+=6*s[iX  ][iY-3];
    	sumYsm+=4*s[iX+1][iY-3];
    	sumYsm+=  s[iX+2][iY-3];

		sumYsm-= 2*s[iX-2][iY-1];
    	sumYsm-= 8*s[iX-1][iY-1];
    	sumYsm-=12*s[iX  ][iY-1];
    	sumYsm-= 8*s[iX+1][iY-1];
    	sumYsm-= 2*s[iX+2][iY-1];
    	
    	sumYsm+=  s[iX-2][iY+1];
    	sumYsm+=4*s[iX-1][iY+1];
    	sumYsm+=6*s[iX  ][iY+1];
    	sumYsm+=4*s[iX+1][iY+1];
    	sumYsm+=  s[iX+2][iY+1];
// Y+1 second 
    	sumYss+=  s[iX-2][iY-1];
    	sumYss+=4*s[iX-1][iY-1];
    	sumYss+=6*s[iX  ][iY-1];
    	sumYss+=4*s[iX+1][iY-1];
    	sumYss+=  s[iX+2][iY-1];

		sumYss-= 2*s[iX-2][iY+1];
    	sumYss-= 8*s[iX-1][iY+1];
    	sumYss-=12*s[iX  ][iY+1];
    	sumYss-= 8*s[iX+1][iY+1];
    	sumYss-= 2*s[iX+2][iY+1];
    	
    	sumYss+=  s[iX-2][iY+3];
    	sumYss+=4*s[iX-1][iY+3];
    	sumYss+=6*s[iX  ][iY+3];
    	sumYss+=4*s[iX+1][iY+3];
    	sumYss+=  s[iX+2][iY+3];
// X first
		sumXf-=  s[iX-1][iY-1];
    	sumXf-=2*s[iX-1][iY  ];
    	sumXf-=  s[iX-1][iY+1];
    	sumXf+=  s[iX+1][iY-1];
    	sumXf+=2*s[iX+1][iY  ];
    	sumXf+=  s[iX+1][iY+1];
// X second
    	sumXs+=  s[iX-2][iY-2];
    	sumXs+=4*s[iX-2][iY-1];
    	sumXs+=6*s[iX-2][iY  ];
    	sumXs+=4*s[iX-2][iY+1];
    	sumXs+=  s[iX-2][iY+2];

		sumXs-= 2*s[iX][iY-2];
    	sumXs-= 8*s[iX][iY-1];
    	sumXs-=12*s[iX][iY  ];
    	sumXs-= 8*s[iX][iY+1];
    	sumXs-= 2*s[iX][iY+2];
    	
    	sumXs+=  s[iX+2][iY-2];
    	sumXs+=4*s[iX+2][iY-1];
    	sumXs+=6*s[iX+2][iY  ];
    	sumXs+=4*s[iX+2][iY+1];
    	sumXs+=  s[iX+2][iY+2];
 // X-1 second   	
    	sumXsm+=  s[iX-3][iY-2];
    	sumXsm+=4*s[iX-3][iY-1];
    	sumXsm+=6*s[iX-3][iY  ];
    	sumXsm+=4*s[iX-3][iY+1];
    	sumXsm+=  s[iX-3][iY+2];

		sumXsm-= 2*s[iX-1][iY-2];
    	sumXsm-= 8*s[iX-1][iY-1];
    	sumXsm-=12*s[iX-1][iY  ];
    	sumXsm-= 8*s[iX-1][iY+1];
    	sumXsm-= 2*s[iX-1][iY+2];
    	
    	sumXsm+=  s[iX+1][iY-2];
    	sumXsm+=4*s[iX+1][iY-1];
    	sumXsm+=6*s[iX+1][iY  ];
    	sumXsm+=4*s[iX+1][iY+1];
    	sumXsm+=  s[iX+1][iY+2];
// X+1 second
    	sumXss+=  s[iX-1][iY-2];
    	sumXss+=4*s[iX-1][iY-1];
    	sumXss+=6*s[iX-1][iY  ];
    	sumXss+=4*s[iX-1][iY+1];
    	sumXss+=  s[iX-1][iY+2];

		sumXss-= 2*s[iX+1][iY-2];
    	sumXss-= 8*s[iX+1][iY-1];
    	sumXss-=12*s[iX+1][iY  ];
    	sumXss-= 8*s[iX+1][iY+1];
    	sumXss-= 2*s[iX+1][iY+2];
    	
    	sumXss+=  s[iX+3][iY-2];
    	sumXss+=4*s[iX+3][iY-1];
    	sumXss+=6*s[iX+3][iY  ];
    	sumXss+=4*s[iX+3][iY+1];
    	sumXss+=  s[iX+3][iY+2];
    	
    	/*int value = sumXs/5+128;
    	if (value > 255)
   			out[y*cols+x] = 255;
   		else if (value < 0)
   			out[y*cols+x] = 0;
   		else
   			out[y*cols+x] = value; 
		*/
		
		if(sumXf > th || sumXf < -th){
			if(sumXs == 0) {
				out[y*cols+x] = 255;
				return;
			}
			if(!SameSign_GPU(&sumXs, &sumXss)){
				if((sumXs+sumXss > 0 && sumXs < 0) ||  (sumXs+sumXss < 0 && sumXs > 0)){
					out[y*cols+x] = 255;
					return;
				}
			}
			if(!SameSign_GPU(&sumXsm, &sumXs)){
				if((sumXsm+sumXs < 0 && sumXs > 0) ||  (sumXs+sumXsm > 0 && sumXs < 0)){
					out[y*cols+x] = 255;
					return;
				}
			}
		}		
		if(sumYf > th || sumYf < -th){
			if(sumYs == 0){
				out[y*cols+x] = 255;
				return;
			}
			if(!SameSign_GPU(&sumYs, &sumYss)){
				if((sumYs+sumYss > 0 && sumYs < 0) ||  (sumYs+sumYss < 0 && sumYs > 0)){
					out[y*cols+x] = 255;
					return;
				}
			}
			if(!SameSign_GPU(&sumYs, &sumYsm)){
				if((sumYsm+sumYs < 0 && sumYs > 0) ||  (sumYs+sumYsm > 0 && sumYs < 0)){
					out[y*cols+x] = 255;
					return;
				}
			}
		}
		else
			out[y*cols+x] = 0;
		
}

__global__ void edgeDetect2x_GPU (const int rows, const int cols, unsigned char *img, unsigned char *out1, unsigned char *out2, int th1, int th2){

    	__shared__ unsigned char s[22][22];
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+3;
		int iY = threadIdx.y+3;
    	
    	if(x < 0 || x >= cols-1 || y < 0 || y >= rows-1) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 6){
    		if(x < 3) s[iX-3][iY] = 0;
    		else{
    			s[iX-3][iY] = img[y*cols+x-3];
    			if(y < 3) s[iX-3][iY-3] = 0;
    			else s[iX-3][iY-3] = img[(y-3)*cols+(x-3)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-3) s[iX+3][iY] = 0;
    		else{
    			s[iX+3][iY] = img[y*cols+(x+3)];
    			if(y >= rows-3) s[iX+3][iY+3] = 0;
    			else s[iX+3][iY+3] = img[(y+3)*cols+x+3];
    		}
    	}
    	if(iY < 6){
    		if(y < 3) s[iX][iY-3] = 0;
    		else{
    			s[iX][iY-3] = img[(y-3)*cols+x];
    			if(x >= cols-3) s[iX+3][iY-3] = 0;
    			else s[iX+3][iY-3] = img[(y-3)*cols+x+3];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-3) s[iX][iY+3] = 0;
    		else{
    			s[iX][iY+3] = img[(y+3)*cols+x];
    			if(x < 3) s[iX-3][iY+3] = 0;
    			else s[iX-3][iY+3] = img[(y+3)*cols+(x-3)];
    		}
    	}
    	
    	__syncthreads();
    		
    	//if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	int sumYf = 0;
    	int sumYs = 0; 
    	int sumYss = 0;   
    	int sumYsm = 0;   	
    	int sumXf = 0;
    	int sumXs = 0;
    	int sumXss = 0;
    	int sumXsm = 0;
  
// Y first		
    	sumYf-=  s[iX-1][iY-1];
    	sumYf-=2*s[iX  ][iY-1];
    	sumYf-=  s[iX+1][iY-1];
    	sumYf+=  s[iX-1][iY+1];
    	sumYf+=2*s[iX  ][iY+1];
    	sumYf+=  s[iX+1][iY+1];
// Y second    	
    	sumYs+=  s[iX-2][iY-2];
    	sumYs+=4*s[iX-1][iY-2];
    	sumYs+=6*s[iX  ][iY-2];
    	sumYs+=4*s[iX+1][iY-2];
    	sumYs+=  s[iX+2][iY-2];

		sumYs-= 2*s[iX-2][iY];
    	sumYs-= 8*s[iX-1][iY];
    	sumYs-=12*s[iX  ][iY];
    	sumYs-= 8*s[iX+1][iY];
    	sumYs-= 2*s[iX+2][iY];
    	
    	sumYs+=  s[iX-2][iY+2];
    	sumYs+=4*s[iX-1][iY+2];
    	sumYs+=6*s[iX  ][iY+2];
    	sumYs+=4*s[iX+1][iY+2];
    	sumYs+=  s[iX+2][iY+2];
// Y-1 second  	
		sumYsm+=  s[iX-2][iY-3];
    	sumYsm+=4*s[iX-1][iY-3];
    	sumYsm+=6*s[iX  ][iY-3];
    	sumYsm+=4*s[iX+1][iY-3];
    	sumYsm+=  s[iX+2][iY-3];

		sumYsm-= 2*s[iX-2][iY-1];
    	sumYsm-= 8*s[iX-1][iY-1];
    	sumYsm-=12*s[iX  ][iY-1];
    	sumYsm-= 8*s[iX+1][iY-1];
    	sumYsm-= 2*s[iX+2][iY-1];
    	
    	sumYsm+=  s[iX-2][iY+1];
    	sumYsm+=4*s[iX-1][iY+1];
    	sumYsm+=6*s[iX  ][iY+1];
    	sumYsm+=4*s[iX+1][iY+1];
    	sumYsm+=  s[iX+2][iY+1];
// Y+1 second 
    	sumYss+=  s[iX-2][iY-1];
    	sumYss+=4*s[iX-1][iY-1];
    	sumYss+=6*s[iX  ][iY-1];
    	sumYss+=4*s[iX+1][iY-1];
    	sumYss+=  s[iX+2][iY-1];

		sumYss-= 2*s[iX-2][iY+1];
    	sumYss-= 8*s[iX-1][iY+1];
    	sumYss-=12*s[iX  ][iY+1];
    	sumYss-= 8*s[iX+1][iY+1];
    	sumYss-= 2*s[iX+2][iY+1];
    	
    	sumYss+=  s[iX-2][iY+3];
    	sumYss+=4*s[iX-1][iY+3];
    	sumYss+=6*s[iX  ][iY+3];
    	sumYss+=4*s[iX+1][iY+3];
    	sumYss+=  s[iX+2][iY+3];
// X first
		sumXf-=  s[iX-1][iY-1];
    	sumXf-=2*s[iX-1][iY  ];
    	sumXf-=  s[iX-1][iY+1];
    	sumXf+=  s[iX+1][iY-1];
    	sumXf+=2*s[iX+1][iY  ];
    	sumXf+=  s[iX+1][iY+1];
// X second
    	sumXs+=  s[iX-2][iY-2];
    	sumXs+=4*s[iX-2][iY-1];
    	sumXs+=6*s[iX-2][iY  ];
    	sumXs+=4*s[iX-2][iY+1];
    	sumXs+=  s[iX-2][iY+2];

		sumXs-= 2*s[iX][iY-2];
    	sumXs-= 8*s[iX][iY-1];
    	sumXs-=12*s[iX][iY  ];
    	sumXs-= 8*s[iX][iY+1];
    	sumXs-= 2*s[iX][iY+2];
    	
    	sumXs+=  s[iX+2][iY-2];
    	sumXs+=4*s[iX+2][iY-1];
    	sumXs+=6*s[iX+2][iY  ];
    	sumXs+=4*s[iX+2][iY+1];
    	sumXs+=  s[iX+2][iY+2];
 // X-1 second   	
    	sumXsm+=  s[iX-3][iY-2];
    	sumXsm+=4*s[iX-3][iY-1];
    	sumXsm+=6*s[iX-3][iY  ];
    	sumXsm+=4*s[iX-3][iY+1];
    	sumXsm+=  s[iX-3][iY+2];

		sumXsm-= 2*s[iX-1][iY-2];
    	sumXsm-= 8*s[iX-1][iY-1];
    	sumXsm-=12*s[iX-1][iY  ];
    	sumXsm-= 8*s[iX-1][iY+1];
    	sumXsm-= 2*s[iX-1][iY+2];
    	
    	sumXsm+=  s[iX+1][iY-2];
    	sumXsm+=4*s[iX+1][iY-1];
    	sumXsm+=6*s[iX+1][iY  ];
    	sumXsm+=4*s[iX+1][iY+1];
    	sumXsm+=  s[iX+1][iY+2];
// X+1 second
    	sumXss+=  s[iX-1][iY-2];
    	sumXss+=4*s[iX-1][iY-1];
    	sumXss+=6*s[iX-1][iY  ];
    	sumXss+=4*s[iX-1][iY+1];
    	sumXss+=  s[iX-1][iY+2];

		sumXss-= 2*s[iX+1][iY-2];
    	sumXss-= 8*s[iX+1][iY-1];
    	sumXss-=12*s[iX+1][iY  ];
    	sumXss-= 8*s[iX+1][iY+1];
    	sumXss-= 2*s[iX+1][iY+2];
    	
    	sumXss+=  s[iX+3][iY-2];
    	sumXss+=4*s[iX+3][iY-1];
    	sumXss+=6*s[iX+3][iY  ];
    	sumXss+=4*s[iX+3][iY+1];
    	sumXss+=  s[iX+3][iY+2];
    	
    	/*int value = sumXs/5+128;
    	if (value > 255)
   			out[y*cols+x] = 255;
   		else if (value < 0)
   			out[y*cols+x] = 0;
   		else
   			out[y*cols+x] = value; 
		*/
		if(sumXf > th1 || sumXf < -th1 || sumYf > th1 || sumYf < -th1){
			if(sumXf > th1 || sumXf < -th1){
				if(sumXs == 0) {
					out1[y*cols+x] = 255;
				}
				if(!SameSign_GPU(&sumXs, &sumXss)){
					if((sumXs+sumXss > 0 && sumXs < 0) ||  (sumXs+sumXss < 0 && sumXs > 0)){
						out1[y*cols+x] = 255;
					}
				}
				if(!SameSign_GPU(&sumXsm, &sumXs)){
					if((sumXsm+sumXs < 0 && sumXs > 0) ||  (sumXs+sumXsm > 0 && sumXs < 0)){
						out1[y*cols+x] = 255;
					}
				}
			}		
			if(sumYf > th1 || sumYf < -th1){
				if(sumYs == 0){
					out1[y*cols+x] = 255;
				}
				if(!SameSign_GPU(&sumYs, &sumYss)){
					if((sumYs+sumYss > 0 && sumYs < 0) ||  (sumYs+sumYss < 0 && sumYs > 0)){
						out1[y*cols+x] = 255;
					}
				}
				if(!SameSign_GPU(&sumYs, &sumYsm)){
					if((sumYsm+sumYs < 0 && sumYs > 0) ||  (sumYs+sumYsm > 0 && sumYs < 0)){
						out1[y*cols+x] = 255;
					}
				}
			}
		}
		else
			out1[y*cols+x] = 0;
			
		if(sumXf > th2 || sumXf < -th2 || sumYf > th2 || sumYf < -th2){
			if(sumXf > th2 || sumXf < -th2){
				if(sumXs == 0) {
					out2[y*cols+x] = 255;
				}
				if(!SameSign_GPU(&sumXs, &sumXss)){
					if((sumXs+sumXss > 0 && sumXs < 0) ||  (sumXs+sumXss < 0 && sumXs > 0)){
						out2[y*cols+x] = 255;
					}
				}
				if(!SameSign_GPU(&sumXsm, &sumXs)){
					if((sumXsm+sumXs < 0 && sumXs > 0) ||  (sumXs+sumXsm > 0 && sumXs < 0)){
						out2[y*cols+x] = 255;
					}
				}
			}		
			if(sumYf > th2 || sumYf < -th2){
				if(sumYs == 0){
					out2[y*cols+x] = 255;
					return;
				}
				if(!SameSign_GPU(&sumYs, &sumYss)){
					if((sumYs+sumYss > 0 && sumYs < 0) ||  (sumYs+sumYss < 0 && sumYs > 0)){
						out2[y*cols+x] = 255;
					}
				}
				if(!SameSign_GPU(&sumYs, &sumYsm)){
					if((sumYsm+sumYs < 0 && sumYs > 0) ||  (sumYs+sumYsm > 0 && sumYs < 0)){
						out2[y*cols+x] = 255;
					}
				}
			}
		}
			else
				out2[y*cols+x] = 0;
		
}

__global__ void prewittYsec_GPU (const int rows, const int cols, int *img, int *des, const int mode){
		
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x <= 2 || x >= cols-2 || y <= 2 || y >= rows-2) return;
    	
    	int sumY = 0;
    	
    	sumY+=img[(y-2)*cols+(x-2)];
    	sumY+=4*img[(y-2)*cols+(x-1)];
    	sumY+=8*img[(y-2)*cols+(x)];
    	sumY+=4*img[(y-2)*cols+(x+1)];
    	sumY+=img[(y-2)*cols+(x+2)];

		sumY-=2*img[(y)*cols+(x-2)];
    	sumY-=8*img[(y)*cols+(x-1)];
    	sumY-=12*img[(y)*cols+(x)];
    	sumY-=8*img[(y)*cols+(x+1)];
    	sumY-=2*img[(y)*cols+(x+2)];
    	
    	sumY+=img[(y+2)*cols+(x-2)];
    	sumY+=4*img[(y+2)*cols+(x-1)];
    	sumY+=8*img[(y+2)*cols+(x)];
    	sumY+=4*img[(y+2)*cols+(x+1)];
    	sumY+=img[(y+2)*cols+(x+2)];

		if(mode == 0) 
			des[y*cols+x] = sumY;
		if(mode == 2)
			des[y*cols+x] = abs(sumY);	
}

__global__ void rotate_GPU (const int rows, const int cols, unsigned char *img, unsigned char *des, float deg){
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

__global__ void blend_GPU (const int rows, const int cols, int *img1, int *img2, int *dest, const float blend, const float scale){
	int x = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x < 0  || x > cols || y < 0 || y > rows) return;
	
	float value = ((scale*blend*img1[y*cols+x])+(scale*(1-blend)*img2[y*cols+x])); 

	dest[y*cols+x] = value;
}

__global__ void blend_GPU (const int rows, const int cols, unsigned char *img1, unsigned char *img2, unsigned char *dest, const float blend, const float scale){
	int x = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x < 0  || x > cols || y < 0 || y > rows) return;
	
	float value = ((scale*blend*img1[y*cols+x])+(scale*(1-blend)*img2[y*cols+x])); 

	dest[y*cols+x] = value;
}

__global__ void edgeDetect(const int rows, const int cols, int *firstX, int *firstY, int *secX, int *secY, int *des){

		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x < 1 || x > cols-1 || y < 1 || y > rows-1) return;
		
		int firstTH = 30;
		//int secTH = 30;
		if(abs(firstX[y*cols+x]) > firstTH ){
			des[y*cols+x] = 0;
			if(!SameSign(secX[y*cols+(x-1)], secX[y*cols+(x+1)]))
				des[y*cols+x] = 255;
				return;
		}		
		else if(abs(firstY[y*cols+x]) > firstTH ){
			des[y*cols+x] = 0;
			if(!SameSign(secY[(y-1)*cols+(x)], secY[(y+1)*cols+(x)]))
				des[y*cols+x] = 255;
				return;
		}		
		else 
			des[y*cols+x] = 0;
}

__global__ void edgeTypeDetect(const int rows, const int cols, unsigned char *img, unsigned char *des){

    	__shared__ unsigned char s[20][20];
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+1;
		int iY = threadIdx.y+1;
    	
    	if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 2){
    		if(x < 1) s[iX-1][iY] = 0;
    		else{
    			s[iX-1][iY] = img[y*cols+x-1];
    			if(y < 1) s[iX-1][iY-1] = 0;
    			else s[iX-1][iY-1] = img[(y-1)*cols+(x-1)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-1) s[iX+1][iY] = 0;
    		else{
    			s[iX+1][iY] = img[y*cols+(x+1)];
    			if(y >= rows-1) s[iX+1][iY+1] = 0;
    			else s[iX+1][iY+1] = img[(y+1)*cols+x+1];
    		}
    	}
    	if(iY < 2){
    		if(y < 1) s[iX][iY-1] = 0;
    		else{
    			s[iX][iY-1] = img[(y-1)*cols+x];
    			if(x >= cols-1) s[iX+1][iY-1] = 0;
    			else s[iX+1][iY-1] = img[(y-1)*cols+x+1];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-1) s[iX][iY+1] = 0;
    		else{
    			s[iX][iY+1] = img[(y+1)*cols+x];
    			if(x < 1) s[iX-1][iY+1] = 0;
    			else s[iX-1][iY+1] = img[(y+1)*cols+(x-1)];
    		}
    	}
    	
		if(s[iX][iY]){
    	
			unsigned char value = 0;
			if(s[iX-1][iY-1]) value |= 0b00000001;
			if(s[iX  ][iY-1]) value |= 0b00000010;
			if(s[iX+1][iY-1]) value |= 0b00000100;
			if(s[iX+1][iY  ]) value |= 0b00001000;
			if(s[iX+1][iY+1]) value |= 0b00010000;
			if(s[iX  ][iY+1]) value |= 0b00100000;
			if(s[iX-1][iY+1]) value |= 0b01000000;
			if(s[iX-1][iY  ]) value |= 0b10000000;
			
			

			if(isNode3(&value)){
				des[y*cols+x] = NODE;
				return;
			}
			if(isVertical(&value)){
				if(isHorizontal(&value)){
					des[y*cols+x] = DIAG;
					return;	
				}
				des[y*cols+x] = VERT;
				return;
			}
			if(isHorizontal(&value)){
				des[y*cols+x] = HORI;
				return;
			}
			if(isEndX(&value)){	
				des[y*cols+x] = END_X;
				return;
			}
			if(isEndY(&value)){	
				des[y*cols+x] = END_Y;
				return;
			}
			if(isEndXXY(&value)){	
				des[y*cols+x] = END_X_XY;
				return;
			}
			if(isEndXYX(&value)){	
				des[y*cols+x] = END_X_YX;
				return;
			}
			if(isEndYXY(&value)){	
				des[y*cols+x] = END_Y_XY;
				return;
			}
			if(isEndYYX(&value)){	
				des[y*cols+x] = END_Y_YX;
				return;
			}
			if(value == 0){
				des[y*cols+x] = NOISE;
				return;
			}
			des[y*cols+x] = 255;
			return;
		}
		des[y*cols+x] = 0;
			
}

__global__ void findNode(const int rows, const int cols, unsigned char *img, unsigned char *des){

    	__shared__ unsigned char s[20][20];
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+2;
		int iY = threadIdx.y+2;
    	
    	if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 4){
    		if(x < 2) s[iX-2][iY] = 0;
    		else{
    			s[iX-2][iY] = img[y*cols+x-2];
    			if(y < 2) s[iX-2][iY-2] = 0;
    			else s[iX-2][iY-2] = img[(y-2)*cols+(x-2)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-2) s[iX+2][iY] = 0;
    		else{
    			s[iX+2][iY] = img[y*cols+(x+2)];
    			if(y >= rows-2) s[iX+2][iY+2] = 0;
    			else s[iX+2][iY+2] = img[(y+2)*cols+x+2];
    		}
    	}
    	if(iY < 4){
    		if(y < 2) s[iX][iY-2] = 0;
    		else{
    			s[iX][iY-2] = img[(y-2)*cols+x];
    			if(x >= cols-2) s[iX+2][iY-2] = 0;
    			else s[iX+2][iY-2] = img[(y-2)*cols+x+2];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-2) s[iX][iY+2] = 0;
    		else{
    			s[iX][iY+2] = img[(y+2)*cols+x];
    			if(x < 2) s[iX-2][iY+2] = 0;
    			else s[iX-2][iY+2] = img[(y+2)*cols+(x-2)];
    		}
    	}
    	
    	__syncthreads();
    	

		
    	if(s[iX][iY]){
    	
			unsigned char value = 0;
			if(s[iX-1][iY-1]) value |= 0b00000001;
			if(s[iX  ][iY-1]) value |= 0b00000010;
			if(s[iX+1][iY-1]) value |= 0b00000100;
			if(s[iX+1][iY  ]) value |= 0b00001000;
			if(s[iX+1][iY+1]) value |= 0b00010000;
			if(s[iX  ][iY+1]) value |= 0b00100000;
			if(s[iX-1][iY+1]) value |= 0b01000000;
			if(s[iX-1][iY  ]) value |= 0b10000000;
			
			if(isNode3(&value)){
				unsigned char xM = 0;
				unsigned char xP = 0;
				unsigned char yM = 0;
				unsigned char yP = 0; 
				if(s[iX-2][iY-1]) xM |= 0b00000001;
				if(s[iX-1][iY-1]) xM |= 0b00000010;
				if(s[iX  ][iY-1]) xM |= 0b00000100;
				if(s[iX  ][iY  ]) xM |= 0b00001000;
				if(s[iX  ][iY+1]) xM |= 0b00010000;
				if(s[iX-1][iY+1]) xM |= 0b00100000;
				if(s[iX-2][iY+1]) xM |= 0b01000000;
				if(s[iX-2][iY  ]) xM |= 0b10000000;

				if(s[iX  ][iY-1]) xP |= 0b00000001;
				if(s[iX+1][iY-1]) xP |= 0b00000010;
				if(s[iX+2][iY-1]) xP |= 0b00000100;
				if(s[iX+2][iY  ]) xP |= 0b00001000;
				if(s[iX+2][iY+1]) xP |= 0b00010000;
				if(s[iX+1][iY+1]) xP |= 0b00100000;
				if(s[iX  ][iY+1]) xP |= 0b01000000;
				if(s[iX  ][iY  ]) xP |= 0b10000000;		
				
				if(s[iX-1][iY-2]) yM |= 0b00000001;
				if(s[iX  ][iY-2]) yM |= 0b00000010;
				if(s[iX+1][iY-2]) yM |= 0b00000100;
				if(s[iX+1][iY-1]) yM |= 0b00001000;
				if(s[iX+1][iY  ]) yM |= 0b00010000;
				if(s[iX  ][iY  ]) yM |= 0b00100000;
				if(s[iX-1][iY  ]) yM |= 0b01000000;
				if(s[iX-1][iY-1]) yM |= 0b10000000;	
				
				if(s[iX-1][iY  ]) yP |= 0b00000001;
				if(s[iX  ][iY  ]) yP |= 0b00000010;
				if(s[iX+1][iY  ]) yP |= 0b00000100;
				if(s[iX+1][iY+1]) yP |= 0b00001000;
				if(s[iX+1][iY+2]) yP |= 0b00010000;
				if(s[iX  ][iY+2]) yP |= 0b00100000;
				if(s[iX-1][iY+2]) yP |= 0b01000000;
				if(s[iX-1][iY+1]) yP |= 0b10000000;
			
				if(!isDeadEnd(&xM) && !isDeadEnd(&xP) && !isDeadEnd(&yM) && !isDeadEnd(&yP))
    				des[y*cols+x] = 255;
    			else
    				des[y*cols+x] = 70;
    		}else 
    			des[y*cols+x] = 70;
    	}else
    		des[y*cols+x] = 0;
    	
}

__global__ void extend(const int rows, const int cols, unsigned char *img, unsigned char *des){

    	__shared__ unsigned char s[20][20];
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+1;
		int iY = threadIdx.y+1;
    	
    	if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 2){
    		if(x < 1) s[iX-1][iY] = 0;
    		else{
    			s[iX-1][iY] = img[y*cols+x-1];
    			if(y < 1) s[iX-1][iY-1] = 0;
    			else s[iX-1][iY-1] = img[(y-1)*cols+(x-1)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-1) s[iX+1][iY] = 0;
    		else{
    			s[iX+1][iY] = img[y*cols+(x+1)];
    			if(y >= rows-1) s[iX+1][iY+1] = 0;
    			else s[iX+1][iY+1] = img[(y+1)*cols+x+1];
    		}
    	}
    	if(iY < 2){
    		if(y < 1) s[iX][iY-1] = 0;
    		else{
    			s[iX][iY-1] = img[(y-1)*cols+x];
    			if(x >= cols-1) s[iX+1][iY-1] = 0;
    			else s[iX+1][iY-1] = img[(y-1)*cols+x+1];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-1) s[iX][iY+1] = 0;
    		else{
    			s[iX][iY+1] = img[(y+1)*cols+x];
    			if(x < 1) s[iX-1][iY+1] = 0;
    			else s[iX-1][iY+1] = img[(y+1)*cols+(x-1)];
    		}
    	}
    	
    	__syncthreads();
    	
    	if(!s[iX][iY]){
    		if(s[iX-1][iY-1] == END_XY || s[iX-1][iY-1] == END_Y_XY || s[iX-1][iY-1] == END_X_XY)
    			des[y*cols+x] = 255;
    		if(s[iX][iY-1] == END_Y || s[iX][iY-1] == END_Y_XY || s[iX][iY-1] == END_Y_YX)
    			des[y*cols+x] = 255;
    		if(s[iX+1][iY-1] == END_YX || s[iX+1][iY-1] == END_X_YX || s[iX+1][iY-1] == END_Y_YX)
    			des[y*cols+x] = 255;
    		if(s[iX+1][iY] == END_X || s[iX+1][iY] == END_X_YX || s[iX+1][iY] == END_X_XY)
    			des[y*cols+x] = 255;
    		if(s[iX+1][iY+1] == END_XY || s[iX+1][iY+1] == END_X_XY || s[iX+1][iY+1] == END_Y_XY)
    			des[y*cols+x] = 255;
    		if(s[iX][iY+1] == END_Y || s[iX][iY+1] == END_Y_XY || s[iX][iY+1] == END_Y_YX)
    			des[y*cols+x] = 255;
    		if(s[iX-1][iY+1] == END_YX || s[iX-1][iY+1] == END_X_XY || s[iX-1][iY+1] == END_Y_XY)
    			des[y*cols+x] = 255;
    		if(s[iX-1][iY] == END_X || s[iX-1][iY] == END_X_XY || s[iX-1][iY] == END_X_YX)
    			des[y*cols+x] = 255;	
    	}
    	
}

__global__ void reduce(const int rows, const int cols, unsigned char *img, unsigned char *des){

    	__shared__ unsigned char s[20][20];
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+1;
		int iY = threadIdx.y+1;
    	
    	if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 2){
    		if(x < 1) s[iX-1][iY] = 0;
    		else{
    			s[iX-1][iY] = img[y*cols+x-1];
    			if(y < 1) s[iX-1][iY-1] = 0;
    			else s[iX-1][iY-1] = img[(y-1)*cols+(x-1)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-1) s[iX+1][iY] = 0;
    		else{
    			s[iX+1][iY] = img[y*cols+(x+1)];
    			if(y >= rows-1) s[iX+1][iY+1] = 0;
    			else s[iX+1][iY+1] = img[(y+1)*cols+x+1];
    		}
    	}
    	if(iY < 2){
    		if(y < 1) s[iX][iY-1] = 0;
    		else{
    			s[iX][iY-1] = img[(y-1)*cols+x];
    			if(x >= cols-1) s[iX+1][iY-1] = 0;
    			else s[iX+1][iY-1] = img[(y-1)*cols+x+1];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-1) s[iX][iY+1] = 0;
    		else{
    			s[iX][iY+1] = img[(y+1)*cols+x];
    			if(x < 1) s[iX-1][iY+1] = 0;
    			else s[iX-1][iY+1] = img[(y+1)*cols+(x-1)];
    		}
    	}
    	
    	__syncthreads();
    	unsigned char value = 0;
    	if(s[iX][iY] >= END_X && s[iX][iY] <= END_Y_YX){
    		if(s[iX-1][iY-1] == VERT || s[iX-1][iY-1] == HORI)
    			value = 255;
    		if(s[iX][iY-1] == VERT)
    			value = 255;
    		if(s[iX+1][iY-1] == VERT || s[iX+1][iY-1] == HORI)
    			value = 255;
    		if(s[iX+1][iY] == HORI)
    			value = 255;
    		if(s[iX+1][iY+1] == VERT || s[iX+1][iY+1] == HORI)
    			value = 255;
    		if(s[iX][iY+1] == VERT)
    			value = 255;
    		if(s[iX-1][iY+1] == VERT || s[iX-1][iY+1] == HORI)
    			value = 255;
    		if(s[iX-1][iY] == HORI)
    			value = 255;	
    		des[y*cols+x] = value;
    	}
    	else if(s[iX][iY] == NOISE)
    		des[y*cols+x] = 0;
    	//else
    		//des[y*cols+x] = 0;
}

__global__ void extender(const int rows, const int cols, unsigned char *low, unsigned char *high, unsigned char *edge){

    	__shared__ unsigned char highS[18][18];
    	__shared__ unsigned char lowS[32][32];
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+8;
		int iY = threadIdx.y+8;
		
		int X = threadIdx.x+1;
		int Y = threadIdx.y+1;
    	
    	if(x < 0 || x >= cols-1 || y < 0 || y >= rows-1) return;
    	
    	highS[X][Y] = high[y*cols+x];
    	
    	lowS[iX][iY] = low[y*cols+x];
    	
		if(X < 2){
    		if(x < 1) highS[X-1][Y] = 0;
    		else{
    			highS[X-1][Y] = high[y*cols+x-1];
    			if(y < 1) highS[X-1][Y-1] = 0;
    			else highS[X-1][Y-1] = high[(y-1)*cols+(x-1)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-1) highS[X+1][Y] = 0;
    		else{
    			highS[X+1][Y] = high[y*cols+(x+1)];
    			if(y >= rows-1) highS[X+1][Y+1] = 0;
    			else highS[X+1][Y+1] = high[(y+1)*cols+x+1];
    		}
    	}
    	if(iY < 2){
    		if(y < 1) highS[X][Y-1] = 0;
    		else{
    			highS[X][Y-1] = high[(y-1)*cols+x];
    			if(x >= cols-1) highS[X+1][Y-1] = 0;
    			else highS[X+1][Y-1] = high[(y-1)*cols+x+1];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-1) highS[X][Y+1] = 0;
    		else{
    			highS[X][Y+1] =high[(y+1)*cols+x];
    			if(x < 1) highS[X-1][Y+1] = 0;
    			else highS[X-1][Y+1] = high[(y+1)*cols+(x-1)];
    		}
    	}
    
    
    	if(iX < 16){
    		if(x < 8) lowS[iX-8][iY] = 0;
    		else{
    			lowS[iX-8][iY] = low[y*cols+x-8];
    			if(y < 8) lowS[iX-8][iY-8] = 0;
    			else lowS[iX-8][iY-8] = low[(y-8)*cols+(x-8)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-8) lowS[iX+8][iY] = 0;
    		else{
    			lowS[iX+8][iY] = low[y*cols+(x+8)];
    			if(y >= rows-8) lowS[iX+8][iY+8] = 0;
    			else lowS[iX+8][iY+8] = low[(y+8)*cols+x+8];
    		}
    	}
    	if(iY < 16){
    		if(y < 8) lowS[iX][iY-8] = 0;
    		else{
    			lowS[iX][iY-8] = low[(y-8)*cols+x];
    			if(x >= cols-8) lowS[iX+8][iY-8] = 0;
    			else lowS[iX+8][iY-8] = low[(y-8)*cols+x+8];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-8) lowS[iX][iY+8] = 0;
    		else{
    			lowS[iX][iY+8] = low[(y+8)*cols+x];
    			if(x < 8) lowS[iX-8][iY+8] = 0;
    			else lowS[iX-8][iY+8] = low[(y+8)*cols+(x-8)];
    		}
    	}
    	
    	__syncthreads();
    	
    	if(highS[X][Y] == 0 || highS[X][Y] == NOISE){
    	//	edge[y*cols+x] = 0;
    		return;
    	}
    	if(highS[X][Y] == VERT || highS[X][Y] == HORI || highS[X][Y] == DIAG || highS[X][Y] == NODE){
    	//	edge[y*cols+x] = highS[X][Y];
    		return;
    	}
    	if(highS[X][Y] == NODE){
    		if((highS[X][Y-1] >= END_X && highS[X][Y-1] <= END_Y_YX)|| (highS[X][Y+1] >= END_X && highS[X][Y+1] <= END_Y_YX)){
    			edge[y*cols+x] = VERT;
    			return;
			} 
			if((highS[X-1][Y] >= END_X && highS[X-1][Y] <= END_Y_YX)|| (highS[X+1][Y] >= END_X && highS[X+1][Y] <= END_Y_YX)){
    			edge[y*cols+x] = VERT;
    			return;
			} 
			if((highS[X-1][Y-1] >= END_X && highS[X-1][Y-1] <= END_Y_YX)|| (highS[X+1][Y-1] >= END_X && highS[X+1][Y-1] <= END_Y_YX)){
    			edge[y*cols+x] = DIAG;
    			return;
			} 
			if((highS[X-1][Y+1] >= END_X && highS[X-1][Y+1] <= END_Y_YX) || (highS[X+1][Y+1] >= END_X && highS[X+1][Y+1] <= END_Y_YX)){
    			edge[y*cols+x] = DIAG;
    			return;
			} 
			return;
    	}
    	__syncthreads();
    	if(highS[X][Y] >= END_X && highS[X][Y] <= END_Y_YX){
    		int xp = 0;
    		int yp = 0;
    		
    		int sourceDirX = 0;
    		int sourceDirY = 0;
    		
    		
    		if(lowS[iX][iY] == VERT || lowS[iX][iY] == HORI || lowS[iX][iY] == DIAG){
    			edge[y*cols+x] = lowS[iX][iY];
    			if(highS[X][Y] == END_X){
    				if(highS[X-1][Y] != 0)	sourceDirX = 1;
    				else sourceDirX = -1;
    			}
    			else if(highS[X][Y] == END_Y){
    				if(highS[X][Y-1] != 0)	sourceDirY = 1;
    				else sourceDirY = -1;
    			}
    			else if(highS[X][Y] == END_XY || highS[X][Y] == END_X_XY || highS[X][Y] == END_Y_XY){
    				if(highS[X-1][Y-1] != 0){
    					sourceDirX = 1;
    					sourceDirY = 1;
    				}else{
    					sourceDirX = -1;
    					sourceDirY = -1;
    				}
    			}
    			else if(highS[X][Y] == END_YX || highS[X][Y] == END_X_YX || highS[X][Y] == END_Y_YX){
    				if(highS[X-1][Y+1] != 0){
    					sourceDirX = 1;
    					sourceDirY = -1;
    				}else{
    					sourceDirX = -1;
    					sourceDirY = +1;
    				}
    			}
    		}else 
    			edge[y*cols+x] = highS[X][Y];
    			
    		int type = 0;
    		while( iX+xp > 0 && iY+yp > 0 && iX+xp < 32 && iY+yp < 32){
    			if(lowS[iX+xp][iY+yp] == HORI || lowS[iX+xp][iY+yp] == VERT || lowS[iX+xp][iY+yp] == DIAG){
    				edge[(y+yp)*cols+(x+xp)] = lowS[iX+xp][iY+yp];
    			}
    			type = lowS[iX+xp][iY+yp];
    			xp+=sourceDirX;
    			yp+=sourceDirY;
    			if(type == HORI){
    				if(lowS[iX+xp][iY+yp] == HORI || lowS[iX+xp][iY+yp] == VERT || lowS[iX+xp][iY+yp] == DIAG){
    					if(sourceDirX > 0){
    						sourceDirX = 1;
    						sourceDirY = 0;
    					}else{
    						sourceDirX = -1;
    						sourceDirY = 0;
    					}
    				}
    				else if(lowS[iX+xp][iY+yp-1] == HORI || lowS[iX+xp][iY+yp-1] == VERT || lowS[iX+xp][iY+yp-1] == DIAG){
    					if(sourceDirX > 0){
    						sourceDirX = 1;
    						sourceDirY = -1;
    					}else{
    						sourceDirX = -1;
    						sourceDirY = -1;
    					}
    				}
    				else if(lowS[iX+xp][iY+yp+1] == HORI || lowS[iX+xp][iY+yp+1] == VERT || lowS[iX+xp][iY+yp+1] == DIAG){
    					if(sourceDirX > 0){
    						sourceDirX = 1;
    						sourceDirY = 1;
    					}else{
    						sourceDirX = -1;
    						sourceDirY = 1;
    					}
    				}
    				
    			}else if(type == VERT){
    				if(lowS[iX+xp][iY+yp] == HORI || lowS[iX+xp][iY+yp] == VERT || lowS[iX+xp][iY+yp] == DIAG){
    					if(sourceDirY > 0){
    						sourceDirX = 0;
    						sourceDirY = 1;
    					}else{
    						sourceDirX = 0;
    						sourceDirY = -1;
    					}
    				}
    				else if(lowS[iX+xp-1][iY+yp] == HORI || lowS[iX+xp-1][iY+yp] == VERT || lowS[iX+xp-1][iY+yp] == DIAG){
    					if(sourceDirY > 0){
    						sourceDirX = -1;
    						sourceDirY = 1;
    					}else{
    						sourceDirX = -1;
    						sourceDirY = -1;
    					}
    				}
    				else if(lowS[iX+xp+1][iY+yp] == HORI || lowS[iX+xp+1][iY+yp] == VERT || lowS[iX+xp+1][iY+yp] == DIAG){
    					if(sourceDirY > 0){
    						sourceDirX = 1;
    						sourceDirY = 1;
    					}else{
    						sourceDirX = 1;
    						sourceDirY = -1;
    					}
    				}
   
    			}else if(type == DIAG){
    				if(sourceDirY == 1){
    					if(sourceDirX == 1){
    						if(lowS[iX+xp][iY+yp] == HORI || lowS[iX+xp][iY+yp] == VERT || lowS[iX+xp][iY+yp] == DIAG){
								sourceDirX = 1;
								sourceDirY = 1;
    						}else if(lowS[iX+xp+1][iY+yp] == HORI || lowS[iX+xp+1][iY+yp] == VERT || lowS[iX+xp+1][iY+yp] == DIAG){
								sourceDirX = 1;
								sourceDirY = 0;
    						}else if(lowS[iX+xp][iY+yp+1] == HORI || lowS[iX+xp][iY+yp+1] == VERT || lowS[iX+xp][iY+yp+1] == DIAG){
								sourceDirX = 0;
								sourceDirY = 1;
    						}
    					}else{
    						if(lowS[iX+xp][iY+yp] == HORI || lowS[iX+xp][iY+yp] == VERT || lowS[iX+xp][iY+yp] == DIAG){
								sourceDirX = -1;
								sourceDirY = 1;
    						}else if(lowS[iX+xp-1][iY+yp] == HORI || lowS[iX+xp-1][iY+yp] == VERT || lowS[iX+xp-1][iY+yp] == DIAG){
								sourceDirX = -1;
								sourceDirY = 0;
    						}else if(lowS[iX+xp][iY+yp+1] == HORI || lowS[iX+xp][iY+yp+1] == VERT || lowS[iX+xp][iY+yp+1] == DIAG){
								sourceDirX = 0;
								sourceDirY = 1;
    						}
    					}
    				}else{
    					if(sourceDirX == 1){
							if(lowS[iX+xp][iY+yp] == HORI || lowS[iX+xp][iY+yp] == VERT || lowS[iX+xp][iY+yp] == DIAG){
								sourceDirX = 1;
								sourceDirY = -1;
    						}else if(lowS[iX+xp+1][iY+yp] == HORI || lowS[iX+xp+1][iY+yp] == VERT || lowS[iX+xp+1][iY+yp] == DIAG){
								sourceDirX = 1;
								sourceDirY = 0;
    						}else if(lowS[iX+xp][iY+yp-1] == HORI || lowS[iX+xp][iY+yp-1] == VERT || lowS[iX+xp][iY+yp-1] == DIAG){
								sourceDirX = 0;
								sourceDirY = -1;
    						}
    					}else{
							if(lowS[iX+xp][iY+yp] == HORI || lowS[iX+xp][iY+yp] == VERT || lowS[iX+xp][iY+yp] == DIAG){
								sourceDirX = -1;
								sourceDirY = -1;
    						}else if(lowS[iX+xp-1][iY+yp] == HORI || lowS[iX+xp-1][iY+yp] == VERT || lowS[iX+xp-1][iY+yp] == DIAG){
								sourceDirX = -1;
								sourceDirY = 0;
    						}else if(lowS[iX+xp][iY+yp-1] == HORI || lowS[iX+xp][iY+yp-1] == VERT || lowS[iX+xp][iY+yp-1] == DIAG){
								sourceDirX = 0;
								sourceDirY = -1;
    						}
    					}
    				}
    			}
    			else {
    				sourceDirX = 0;
    				sourceDirY = 0;
    				break;
    			}
    		}   		
    	}
}



__global__ void difference(const int rows, const int cols, unsigned char *in1, unsigned char *in2, unsigned char *dif){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned char i1 = in1[y*cols+x];
	unsigned char i2 = in2[y*cols+x];
	
	if((i1 == 0 || (i1 >= END_X &&  i1 <= END_Y_YX)) && (i2 == HORI || i2 == VERT || i2 == DIAG)) dif[y*cols+x] = 255;
	else  dif[y*cols+x] = 0;
	
}

__global__ void inprove(const int rows, const int cols, unsigned char *in1, unsigned char *in2, unsigned char *out){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned char i1 = in1[y*cols+x];
	unsigned char i2 = in2[y*cols+x];
	
	if(i1 == HORI || i1 == VERT || i1 == DIAG){
		out[y*cols+x] = i1;
		return;
	}
	if(i2 == HORI || i2 == VERT || i2 == DIAG){
		out[y*cols+x] = i2;
		return;
	}
	if(i2 >= END_X && i2 <= END_Y_YX){
		out[y*cols+x] = i2;
		return;
	}
	if(i1 >= END_X && i1 <= END_Y_YX){
		out[y*cols+x] = i1;
		return;
	}	
	if(i1 == NODE){
		out[y*cols+x] = i1;
		return;
	}
	else  out[y*cols+x] = 0;
	
}















