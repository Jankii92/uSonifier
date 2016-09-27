#include "improc.h"
#include <math.h>
#include <stdlib.h>
#include <curand.h>
#include "cpyToShared.h"
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
#define VERT  70
#define HORI  50
#define DIAG  90
#define NODE  250
#define NODE_U  100
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

__device__ bool isEnd(unsigned char *x){
	if(isEndX(x) || isEndY(x) || isEndXXY(x) || isEndXYX(x) || isEndYXY(x) || isEndYYX(x))
		return true;
	else 
		return false;
}

__device__ bool isNode3(unsigned char *x, int* iX, int* iY,unsigned char  s[20][20], unsigned char *type ){
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
		 ((*x ^ 0b10110110) & 0b11101011))) {
		 	unsigned char value[9];
		 	value[0] = 0;
		 	value[1] = 0;
		 	value[2] = 0;
		 	value[3] = 0;
		 	value[4] = 0;
		 	value[5] = 0;
		 	value[6] = 0;
		 	value[7] = 0;
		 	
		 	if(s[*iX-1-1][*iY-1-1]) value[0] |= 0b00000001;
			if(s[*iX-1  ][*iY-1-1]) value[0] |= 0b00000010;
			if(s[*iX+1-1][*iY-1-1]) value[0] |= 0b00000100;
			if(s[*iX+1-1][*iY-1  ]) value[0] |= 0b00001000;
			if(s[*iX+1-1][*iY+1-1]) value[0] |= 0b00010000;
			if(s[*iX-1  ][*iY+1-1]) value[0] |= 0b00100000;
			if(s[*iX-1-1][*iY+1-1]) value[0] |= 0b01000000;
			if(s[*iX-1-1][*iY-1  ]) value[0] |= 0b10000000;
			
			if(s[*iX-1][*iY-1-1]) value[1] |= 0b00000001;
			if(s[*iX  ][*iY-1-1]) value[1] |= 0b00000010;
			if(s[*iX+1][*iY-1-1]) value[1] |= 0b00000100;
			if(s[*iX+1][*iY-1  ]) value[1] |= 0b00001000;
			if(s[*iX+1][*iY+1-1]) value[1] |= 0b00010000;
			if(s[*iX  ][*iY+1-1]) value[1] |= 0b00100000;
			if(s[*iX-1][*iY+1-1]) value[1] |= 0b01000000;
			if(s[*iX-1][*iY-1  ]) value[1] |= 0b10000000;
			
			if(s[*iX-1+1][*iY-1-1]) value[2] |= 0b00000001;
			if(s[*iX+1  ][*iY-1-1]) value[2] |= 0b00000010;
			if(s[*iX+1+1][*iY-1-1]) value[2] |= 0b00000100;
			if(s[*iX+1+1][*iY-1  ]) value[2] |= 0b00001000;
			if(s[*iX+1+1][*iY+1-1]) value[2] |= 0b00010000;
			if(s[*iX+1  ][*iY+1-1]) value[2] |= 0b00100000;
			if(s[*iX-1+1][*iY+1-1]) value[2] |= 0b01000000;
			if(s[*iX-1+1][*iY-1  ]) value[2] |= 0b10000000;

		 	if(s[*iX-1-1][*iY-1]) value[7] |= 0b00000001;
			if(s[*iX-1  ][*iY-1]) value[7] |= 0b00000010;
			if(s[*iX+1-1][*iY-1]) value[7] |= 0b00000100;
			if(s[*iX+1-1][*iY  ]) value[7] |= 0b00001000;
			if(s[*iX+1-1][*iY+1]) value[7] |= 0b00010000;
			if(s[*iX-1  ][*iY+1]) value[7] |= 0b00100000;
			if(s[*iX-1-1][*iY+1]) value[7] |= 0b01000000;
			if(s[*iX-1-1][*iY  ]) value[7] |= 0b10000000;
			
		 	if(s[*iX-1+1][*iY-1]) value[3] |= 0b00000001;
			if(s[*iX+1  ][*iY-1]) value[3] |= 0b00000010;
			if(s[*iX+1+1][*iY-1]) value[3] |= 0b00000100;
			if(s[*iX+1+1][*iY  ]) value[3] |= 0b00001000;
			if(s[*iX+1+1][*iY+1]) value[3] |= 0b00010000;
			if(s[*iX+1  ][*iY+1]) value[3] |= 0b00100000;
			if(s[*iX-1+1][*iY+1]) value[3] |= 0b01000000;
			if(s[*iX-1+1][*iY  ]) value[3] |= 0b10000000;
			
			if(s[*iX-1-1][*iY-1+1]) value[6] |= 0b00000001;
			if(s[*iX-1  ][*iY-1+1]) value[6] |= 0b00000010;
			if(s[*iX+1-1][*iY-1+1]) value[6] |= 0b00000100;
			if(s[*iX+1-1][*iY+1  ]) value[6] |= 0b00001000;
			if(s[*iX+1-1][*iY+1+1]) value[6] |= 0b00010000;
			if(s[*iX-1  ][*iY+1+1]) value[6] |= 0b00100000;
			if(s[*iX-1-1][*iY+1+1]) value[6] |= 0b01000000;
			if(s[*iX-1-1][*iY+1  ]) value[6] |= 0b10000000;
			
			if(s[*iX-1][*iY-1+1]) value[5] |= 0b00000001;
			if(s[*iX  ][*iY-1+1]) value[5] |= 0b00000010;
			if(s[*iX+1][*iY-1+1]) value[5] |= 0b00000100;
			if(s[*iX+1][*iY+1  ]) value[5] |= 0b00001000;
			if(s[*iX+1][*iY+1+1]) value[5] |= 0b00010000;
			if(s[*iX  ][*iY+1+1]) value[5] |= 0b00100000;
			if(s[*iX-1][*iY+1+1]) value[5] |= 0b01000000;
			if(s[*iX-1][*iY+1  ]) value[5] |= 0b10000000;
			
			if(s[*iX-1+1][*iY-1+1]) value[4] |= 0b00000001;
			if(s[*iX+1  ][*iY-1+1]) value[4] |= 0b00000010;
			if(s[*iX+1+1][*iY-1+1]) value[4] |= 0b00000100;
			if(s[*iX+1+1][*iY+1  ]) value[4] |= 0b00001000;
			if(s[*iX+1+1][*iY+1+1]) value[4] |= 0b00010000;
			if(s[*iX+1  ][*iY+1+1]) value[4] |= 0b00100000;
			if(s[*iX-1+1][*iY+1+1]) value[4] |= 0b01000000;
			if(s[*iX-1+1][*iY+1  ]) value[4] |= 0b10000000;
			
			if(isEnd(&value[0]) ||// isDeadEnd(&value[0]) ||
				isEnd(&value[1]) || //isDeadEnd(&value[1]) ||
				isEnd(&value[2]) || //isDeadEnd(&value[2]) ||
				isEnd(&value[3]) || //isDeadEnd(&value[3]) ||
				isEnd(&value[4]) || //isDeadEnd(&value[4]) ||
				isEnd(&value[5]) || //isDeadEnd(&value[5]) ||
				isEnd(&value[6]) || //isDeadEnd(&value[6]) ||
				isEnd(&value[7]) /*|| isDeadEnd(&value[7])*/){
					*type = NODE_U;
					return true;
				}else{
					*type = NODE;
					return true;
				}	
		 }else {
			return false;
		 }
		
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
    		if(x < 3){
    			s[iX-3][iY] = 0;
    			if(iY < 6) s[iX-3][iY-3] = 0;
    		}else{
    			s[iX-3][iY] = img[y*cols+x-3];
    			if(y < 3) s[iX-3][iY-3] = 0;
    			else s[iX-3][iY-3] = img[(y-3)*cols+(x-3)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-3){
    			s[iX+3][iY] = 0;
    			if(iY >= blockDim.y) s[iX+3][iY+3] = 0;
    		}else{
    			s[iX+3][iY] = img[y*cols+(x+3)];
    			if(y >= rows-3) s[iX+3][iY+3] = 0;
    			else s[iX+3][iY+3] = img[(y+3)*cols+x+3];
    		}
    	}
    	if(iY < 6){
    		if(y < 3){ 
    			s[iX][iY-3] = 0;
    			if(iX >= blockDim.x) s[iX+3][iY-3] = 0;
    		}else{
    			s[iX][iY-3] = img[(y-3)*cols+x];
    			if(x >= cols-3) s[iX+3][iY-3] = 0;
    			else s[iX+3][iY-3] = img[(y-3)*cols+x+3];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-3){
    			s[iX][iY+3] = 0;
    			if(iX < 6) s[iX-3][iY+3] = 0;
    		}else{
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
  		__syncthreads();
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
		
		if(sumXf > th || sumXf < -th || sumYf > th || sumYf < -th){
			if(sumXf > th || sumXf < -th){
				if(sumXs == 0) {
					out[y*cols+x] = 255;
				}
				if(!SameSign_GPU(&sumXs, &sumXss)){
					if((sumXs+sumXss > 0 && sumXs < 0) ||  (sumXs+sumXss < 0 && sumXs > 0)){
						out[y*cols+x] = 255;
					}
				}
				if(!SameSign_GPU(&sumXsm, &sumXs)){
					if((sumXsm+sumXs < 0 && sumXs > 0) ||  (sumXs+sumXsm > 0 && sumXs < 0)){
						out[y*cols+x] = 255;
					}
				}
			}		
			if(sumYf > th || sumYf < -th){
				if(sumYs == 0){
					out[y*cols+x] = 255;
				}
				if(!SameSign_GPU(&sumYs, &sumYss)){
					if((sumYs+sumYss > 0 && sumYs < 0) ||  (sumYs+sumYss < 0 && sumYs > 0)){
						out[y*cols+x] = 255;
					}
				}
				if(!SameSign_GPU(&sumYs, &sumYsm)){
					if((sumYsm+sumYs < 0 && sumYs > 0) ||  (sumYs+sumYsm > 0 && sumYs < 0)){
						out[y*cols+x] = 255;
					}
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
    	
    	if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 6){
    		if(x < 3){
    			s[iX-3][iY] = 0;
    			if(iY < 6) s[iX-3][iY-3] = 0;
    		}else{
    			s[iX-3][iY] = img[y*cols+x-3];
    			if(y < 3) s[iX-3][iY-3] = 0;
    			else s[iX-3][iY-3] = img[(y-3)*cols+(x-3)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-3){
    			s[iX+3][iY] = 0;
    			if(iY >= blockDim.y) s[iX+3][iY+3] = 0;
    		}else{
    			s[iX+3][iY] = img[y*cols+(x+3)];
    			if(y >= rows-3) s[iX+3][iY+3] = 0;
    			else s[iX+3][iY+3] = img[(y+3)*cols+x+3];
    		}
    	}
    	if(iY < 6){
    		if(y < 3){ 
    			s[iX][iY-3] = 0;
    			if(iX >= blockDim.x) s[iX+3][iY-3] = 0;
    		}else{
    			s[iX][iY-3] = img[(y-3)*cols+x];
    			if(x >= cols-3) s[iX+3][iY-3] = 0;
    			else s[iX+3][iY-3] = img[(y-3)*cols+x+3];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-3){
    			s[iX][iY+3] = 0;
    			if(iX < 6) s[iX-3][iY+3] = 0;
    		}else{
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
  		__syncthreads();
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

__global__ void edgeTypeDetectCleanup(const int rows, const int cols, unsigned char *img, unsigned char *des){

    	__shared__ unsigned char s[20][20];
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+2;
		int iY = threadIdx.y+2;
    	
    	if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 4){
    		if(x < 2){ 
    			s[iX-2][iY] = 0;
    			if(iY < 4) s[iX-2][iY-2] = 0;
    		}else{
    			s[iX-2][iY] = img[y*cols+x-2];
    			if(y < 2) s[iX-2][iY-2] = 0;
    			else s[iX-2][iY-2] = img[(y-2)*cols+(x-2)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-2){ 
    			s[iX+2][iY] = 0;
    			if(iY >= blockDim.y) s[iX+2][iY+2] = 0;
    		}else{
    			s[iX+2][iY] = img[y*cols+(x+2)];
    			if(y >= rows-2) s[iX+2][iY+2] = 0;
    			else s[iX+2][iY+2] = img[(y+2)*cols+x+2];
    		}
    	}
    	if(iY < 4){
    		if(y < 2){
    			s[iX][iY-2] = 0;
    			if(iX >= blockDim.x) s[iX+2][iY-2] = 0;
    		}else{
    			s[iX][iY-2] = img[(y-2)*cols+x];
    			if(x >= cols-2) s[iX+2][iY-2] = 0;
    			else s[iX+2][iY-2] = img[(y-2)*cols+x+2];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-2){
    			s[iX][iY+2] = 0;
    			if(iX < 4) s[iX-2][iY+2] = 0;
    		}else{
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
			
			unsigned char k;
			if(isNode3(&value, &iX, &iY, s, &k)){
				des[y*cols+x] = 255;
				return;
			}
			if(isVertical(&value)){
				if(isHorizontal(&value)){
					des[y*cols+x] = 255;
					return;	
				}
				des[y*cols+x] = 255;
				return;
			}
			if(isHorizontal(&value)){
				des[y*cols+x] = 255;
				return;
			}
			des[y*cols+x] = 0;//255;
			return;
		}
		des[y*cols+x] = 0;
			
}

__global__ void edgeTypeDetect(const int rows, const int cols, unsigned char *img, unsigned char *des){

    	__shared__ unsigned char s[20][20];
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+2;
		int iY = threadIdx.y+2;
    	
    	if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	s[iX][iY] = img[y*cols+x];
    	
    
    	if(iX < 4){
    		if(x < 2){ 
    			s[iX-2][iY] = 0;
    			if(iY < 4) s[iX-2][iY-2] = 0;
    		}else{
    			s[iX-2][iY] = img[y*cols+x-2];
    			if(y < 2) s[iX-2][iY-2] = 0;
    			else s[iX-2][iY-2] = img[(y-2)*cols+(x-2)];  		
    		}
    	}
    	if(iX >= blockDim.x){
    		if(x >= cols-2){ 
    			s[iX+2][iY] = 0;
    			if(iY >= blockDim.y) s[iX+2][iY+2] = 0;
    		}else{
    			s[iX+2][iY] = img[y*cols+(x+2)];
    			if(y >= rows-2) s[iX+2][iY+2] = 0;
    			else s[iX+2][iY+2] = img[(y+2)*cols+x+2];
    		}
    	}
    	if(iY < 4){
    		if(y < 2){
    			s[iX][iY-2] = 0;
    			if(iX >= blockDim.x) s[iX+2][iY-2] = 0;
    		}else{
    			s[iX][iY-2] = img[(y-2)*cols+x];
    			if(x >= cols-2) s[iX+2][iY-2] = 0;
    			else s[iX+2][iY-2] = img[(y-2)*cols+x+2];
    		}
    	}
    	if(iY >= blockDim.y){
    		if(y >= rows-2){
    			s[iX][iY+2] = 0;
    			if(iX < 4) s[iX-2][iY+2] = 0;
    		}else{
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
			
			unsigned char k;
			if(isNode3(&value, &iX, &iY, s, &k)){
				des[y*cols+x] = k;
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
				des[y*cols+x] = 0;
				des[y*cols+x] = END_X;
				return;
			}
			if(isEndY(&value)){	
				des[y*cols+x] = 0;
				des[y*cols+x] = END_Y;
				return;
			}
			if(isEndXXY(&value)){	
				des[y*cols+x] = 0;
				des[y*cols+x] = END_X_XY;
				return;
			}
			if(isEndXYX(&value)){
				des[y*cols+x] = 0;	
				des[y*cols+x] = END_X_YX;
				return;
			}
			if(isEndYXY(&value)){
			
				des[y*cols+x] = 0;	
				des[y*cols+x] = END_Y_XY;
				return;
			}
			if(isEndYYX(&value)){
			
				des[y*cols+x] = 0;	
				des[y*cols+x] = END_Y_YX;
				return;
			}
			if(value == 0){
				des[y*cols+x] = NOISE;
				des[y*cols+x] = 0;
				return;
			}
			des[y*cols+x] = 0;//255;
			return;
		}
		des[y*cols+x] = 0;
			
}

/*__global__ void findNode(const int rows, const int cols, unsigned char *img, unsigned char *des){

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
    	
}*/

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
__device__ void findMax(int smem[64], int* maxValue, int* index){
	
	__shared__ unsigned char indexs[32];
	if(threadIdx.x < 32 && threadIdx.y == 0){
		if(smem[threadIdx.x+32] > smem[threadIdx.x]){
			smem[threadIdx.x] = smem[threadIdx.x+32];
			indexs[threadIdx.x] = threadIdx.x+32;
		}else{
			smem[threadIdx.x] = smem[threadIdx.x];
			indexs[threadIdx.x] = threadIdx.x;
		}
	}
	__syncthreads();
	if(threadIdx.x < 16 && threadIdx.y == 0){
		if(smem[threadIdx.x+16] > smem[threadIdx.x]){
			smem[threadIdx.x] = smem[threadIdx.x+16];
			indexs[threadIdx.x] = indexs[threadIdx.x+16];
		}
	}
	__syncthreads();
	if(threadIdx.x < 8 && threadIdx.y == 0){
		if(smem[threadIdx.x+8] > smem[threadIdx.x]){
			smem[threadIdx.x] = smem[threadIdx.x+8];
			indexs[threadIdx.x] = indexs[threadIdx.x+8];
		}
	}
	__syncthreads();
	if(threadIdx.x < 4 && threadIdx.y == 0){
		if(smem[threadIdx.x+4] > smem[threadIdx.x]){
			smem[threadIdx.x] = smem[threadIdx.x+4];
			indexs[threadIdx.x] = indexs[threadIdx.x+4];
		}
	}
	__syncthreads();
	if(threadIdx.x < 2 && threadIdx.y == 0){
		if(smem[threadIdx.x+2] > smem[threadIdx.x]){
			smem[threadIdx.x] = smem[threadIdx.x+2];
			indexs[threadIdx.x] = indexs[threadIdx.x+2];
		}
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y == 0){
		if(smem[threadIdx.x+1] > smem[threadIdx.x]){
			smem[threadIdx.x] = smem[threadIdx.x+1];
			indexs[threadIdx.x] = indexs[threadIdx.x+1];
		}
	}
	__syncthreads();
	
	if(threadIdx.x < 1 && threadIdx.y == 0){
		(*maxValue) = smem[0];
		(*index) = indexs[0];
		//smem[indexs[0]] = 0;
	}
	__syncthreads();
}

__device__ void findMaxFast(int smem[48][16], int* maxValue, int* index){
	
	__shared__ unsigned char indexs[24];
	if(threadIdx.x < 24 && threadIdx.y == 0){
		if(smem[threadIdx.x+24][0] > smem[threadIdx.x][0]){
			smem[threadIdx.x][0] = smem[threadIdx.x+24][0];
			indexs[threadIdx.x] = threadIdx.x+24;
		}else{
			smem[threadIdx.x][0] = smem[threadIdx.x][0];
			indexs[threadIdx.x] = threadIdx.x;
		}
	}
	__syncthreads();
	if(threadIdx.x < 16 && threadIdx.y == 0){
		if(smem[threadIdx.x+16][0] > smem[threadIdx.x][0]){
			smem[threadIdx.x][0] = smem[threadIdx.x+16][0];
			indexs[threadIdx.x] = indexs[threadIdx.x+16];
		}
	}
	__syncthreads();
	if(threadIdx.x < 8 && threadIdx.y == 0){
		if(smem[threadIdx.x+8][0] > smem[threadIdx.x][0]){
			smem[threadIdx.x][0] = smem[threadIdx.x+8][0];
			indexs[threadIdx.x] = indexs[threadIdx.x+8];
		}
	}
	__syncthreads();
	if(threadIdx.x < 4 && threadIdx.y == 0){
		if(smem[threadIdx.x+4][0] > smem[threadIdx.x][0]){
			smem[threadIdx.x][0] = smem[threadIdx.x+4][0];
			indexs[threadIdx.x] = indexs[threadIdx.x+4];
		}
	}
	__syncthreads();
	if(threadIdx.x < 2 && threadIdx.y == 0){
		if(smem[threadIdx.x+2][0] > smem[threadIdx.x][0]){
			smem[threadIdx.x][0] = smem[threadIdx.x+2][0];
			indexs[threadIdx.x] = indexs[threadIdx.x+2];
		}
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y == 0){
		if(smem[threadIdx.x+1][0] > smem[threadIdx.x][0]){
			smem[threadIdx.x][0] = smem[threadIdx.x+1][0];
			indexs[threadIdx.x] = indexs[threadIdx.x+1];
		}
	}
	__syncthreads();
	
	if(threadIdx.x < 1 && threadIdx.y == 0){
		(*maxValue) = smem[0][0];
		(*index) = indexs[0];
	}
	__syncthreads();
}

__device__ void findMaxFastDouble(int s1[48][16], int s2[48][16], int* index1, int* maxValue1, int* index2 , int* maxValue2){
	
	__shared__ unsigned char indexs1[24];
	__shared__ unsigned char indexs2[24];
	if(threadIdx.x < 24 && threadIdx.y == 0){
		if(s1[threadIdx.x+24][0] > s1[threadIdx.x][0]){
			s1[threadIdx.x][0] = s1[threadIdx.x+24][0];
			indexs1[threadIdx.x] = threadIdx.x+24;
		}else{
			indexs1[threadIdx.x] = threadIdx.x;
		}
	}
	if(threadIdx.x < 24 && threadIdx.y == 1){
		if(s2[threadIdx.x+24][0] > s2[threadIdx.x][0]){
			s2[threadIdx.x][0] = s2[threadIdx.x+24][0];
			indexs2[threadIdx.x] = threadIdx.x+24;
		}else{
			indexs2[threadIdx.x] = threadIdx.x;
		}
	}
	__syncthreads();
	if(threadIdx.x < 16 && threadIdx.y == 0){
		if(s1[threadIdx.x+16][0] > s1[threadIdx.x][0]){
			s1[threadIdx.x][0] = s1[threadIdx.x+16][0];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+16];
		}
	}
	if(threadIdx.x < 16 && threadIdx.y == 1){
		if(s2[threadIdx.x+16][0] > s2[threadIdx.x][0]){
			s2[threadIdx.x][0] = s2[threadIdx.x+16][0];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+16];
		}
	}
	__syncthreads();
	if(threadIdx.x < 8 && threadIdx.y == 0){
		if(s1[threadIdx.x+8][0] > s1[threadIdx.x][0]){
			s1[threadIdx.x][0] = s1[threadIdx.x+8][0];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+8];
		}
	}
	if(threadIdx.x < 8 && threadIdx.y == 1){
		if(s2[threadIdx.x+8][0] > s2[threadIdx.x][0]){
			s2[threadIdx.x][0] = s2[threadIdx.x+8][0];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+8];
		}
	}
	__syncthreads();
	if(threadIdx.x < 4 && threadIdx.y == 0){
		if(s1[threadIdx.x+4][0] > s1[threadIdx.x][0]){
			s1[threadIdx.x][0] = s1[threadIdx.x+4][0];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+4];
		}
	}	
	if(threadIdx.x < 4 && threadIdx.y == 1){
		if(s2[threadIdx.x+4][0] > s2[threadIdx.x][0]){
			s2[threadIdx.x][0] = s2[threadIdx.x+4][0];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+4];
		}
	}
	__syncthreads();
	if(threadIdx.x < 2 && threadIdx.y == 0){
		if(s1[threadIdx.x+2][0] > s1[threadIdx.x][0]){
			s1[threadIdx.x][0] = s1[threadIdx.x+2][0];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+2];
		}
	}
	if(threadIdx.x < 2 && threadIdx.y == 1){
		if(s2[threadIdx.x+2][0] > s2[threadIdx.x][0]){
			s2[threadIdx.x][0] = s2[threadIdx.x+2][0];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+2];
		}
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y == 0){
		if(s1[threadIdx.x+1][0] > s1[threadIdx.x][0]){
			s1[threadIdx.x][0] = s1[threadIdx.x+1][0];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+1];
		}
	}
	if(threadIdx.x < 1 && threadIdx.y == 1){
		if(s2[threadIdx.x+1][0] > s2[threadIdx.x][0]){
			s2[threadIdx.x][0] = s2[threadIdx.x+1][0];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+1];
		}
	}
	__syncthreads();
	
	if(threadIdx.x < 1 && threadIdx.y == 0){
		(*maxValue1) = s1[0][0];
		(*index1) = indexs1[0];
	}
	if(threadIdx.x < 1 && threadIdx.y == 1){
		(*maxValue2) = s2[0][0];
		(*index2) = indexs2[0];
	}
	__syncthreads();
}

__device__ void findMaxFastDoubleCpy(int s1[48][16], int s2[48][16], int* index1, int* maxValue1, int* index2 , int* maxValue2, unsigned char cpy1[24], unsigned char cpy2[24]){
	
	__shared__ unsigned char indexs1[24];
	__shared__ unsigned char indexs2[24];
	if(threadIdx.y == 0){
		if(s1[threadIdx.x][0] < 0) s1[threadIdx.x][0] = 0;
		if(s2[threadIdx.x][0] < 0) s2[threadIdx.x][0] = 0;
		if(s1[threadIdx.x][0] > 255) s1[threadIdx.x][0] = 255;
		if(s2[threadIdx.x][0] > 255) s2[threadIdx.x][0] = 255;
	}
	__syncthreads();
	if(threadIdx.x < 24 && threadIdx.y == 0){
		if(s1[threadIdx.x+24][0] > s1[threadIdx.x][0]){
			cpy1[threadIdx.x] = s1[threadIdx.x+24][0];
			indexs1[threadIdx.x] = threadIdx.x+24;
		}else{
			cpy1[threadIdx.x] = s1[threadIdx.x][0];
			indexs1[threadIdx.x] = threadIdx.x;
		}
	}
	if(threadIdx.x < 24 && threadIdx.y == 1){
		if(s2[threadIdx.x+24][0] > s2[threadIdx.x][0]){
			cpy2[threadIdx.x] = s2[threadIdx.x+24][0];
			indexs2[threadIdx.x] = threadIdx.x+24;
		}else{
			cpy2[threadIdx.x] = s2[threadIdx.x][0];
			indexs2[threadIdx.x] = threadIdx.x;
		}
	}
	__syncthreads();
	if(threadIdx.x < 12 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+12] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+12];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+12];
		}
	}
	if(threadIdx.x < 12 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+12] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+12];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+12];
		}
	}
	__syncthreads();
	if(threadIdx.x < 8 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+8] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+8];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+8];
		}
	}
	if(threadIdx.x < 8 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+8] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+8];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+8];
		}
	}
	__syncthreads();
	if(threadIdx.x < 4 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+4] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+4];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+4];
		}
	}	
	if(threadIdx.x < 4 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+4] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+4];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+4];
		}
	}
	__syncthreads();
	if(threadIdx.x < 2 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+2] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+2];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+2];
		}
	}
	if(threadIdx.x < 2 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+2] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+2];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+2];
		}
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+1] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+1];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+1];
		}
	}
	if(threadIdx.x < 1 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+1] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+1];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+1];
		}
	}
	__syncthreads();
	
	if(threadIdx.x < 1 && threadIdx.y == 0){
		(*maxValue1) = cpy1[0];
		(*index1) = indexs1[0];
		s1[indexs1[0]][0] = 0;
		if((*index1) > 0 ){
			s1[indexs1[0]-1][0] = 0;
			//s1[indexs1[0]-2][0] = 0;
		}
		if((*index1) > 1 ){
			s1[indexs1[0]-1][0] = 0;
			s1[indexs1[0]-2][0] = 0;
		} 	 		
		if((*index1) < 46){
			s1[indexs1[0]+1][0] = 0;
			s1[indexs1[0]+2][0] = 0;
		} 
		if((*index1) < 47){
			s1[indexs1[0]+1][0] = 0;
			//s1[indexs1[0]+2][0] = 0;
		} 
	}
	if(threadIdx.x < 1 && threadIdx.y == 1){
		(*maxValue2) = cpy2[0];
		(*index2) = indexs2[0];
		s2[indexs2[0]][0] = 0;
		if((*index2) > 0 ){
			s2[indexs2[0]-1][0] = 0;
			//s1[indexs1[0]-2][0] = 0;
		}
		if((*index2) > 1 ){
			s2[indexs2[0]-1][0] = 0;
			s2[indexs2[0]-2][0] = 0;
		} 	 		
		if((*index2) < 46){
			s2[indexs2[0]+1][0] = 0;
			s2[indexs1[0]+2][0] = 0;
		} 
		if((*index2) < 47){
			s2[indexs2[0]+1][0] = 0;
			//s1[indexs1[0]+2][0] = 0;
		} 
	}
	__syncthreads();
	
	if(threadIdx.x < 24 && threadIdx.y == 0){
		if(s1[threadIdx.x+24][0] > s1[threadIdx.x][0]){
			cpy1[threadIdx.x] = s1[threadIdx.x+24][0];
			indexs1[threadIdx.x] = threadIdx.x+24;
		}else{
			cpy1[threadIdx.x] = s1[threadIdx.x][0];
			indexs1[threadIdx.x] = threadIdx.x;
		}
	}
	if(threadIdx.x < 24 && threadIdx.y == 1){
		if(s2[threadIdx.x+24][0] > s2[threadIdx.x][0]){
			cpy2[threadIdx.x] = s2[threadIdx.x+24][0];
			indexs2[threadIdx.x] = threadIdx.x+24;
		}else{
			cpy2[threadIdx.x] = s2[threadIdx.x][0];
			indexs2[threadIdx.x] = threadIdx.x;
		}
	}
	__syncthreads();
	if(threadIdx.x < 12 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+12] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+12];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+12];
		}
	}
	if(threadIdx.x < 12 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+12] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+12];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+12];
		}
	}
	__syncthreads();
	if(threadIdx.x < 8 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+8] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+8];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+8];
		}
	}
	if(threadIdx.x < 8 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+8] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+8];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+8];
		}
	}
	__syncthreads();
	if(threadIdx.x < 4 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+4] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+4];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+4];
		}
	}	
	if(threadIdx.x < 4 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+4] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+4];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+4];
		}
	}
	__syncthreads();
	if(threadIdx.x < 2 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+2] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+2];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+2];
		}
	}
	if(threadIdx.x < 2 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+2] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+2];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+2];
		}
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y == 0){
		if(cpy1[threadIdx.x+1] > cpy1[threadIdx.x]){
			cpy1[threadIdx.x] = cpy1[threadIdx.x+1];
			indexs1[threadIdx.x] = indexs1[threadIdx.x+1];
		}
	}
	if(threadIdx.x < 1 && threadIdx.y == 1){
		if(cpy2[threadIdx.x+1] > cpy2[threadIdx.x]){
			cpy2[threadIdx.x] = cpy2[threadIdx.x+1];
			indexs2[threadIdx.x] = indexs2[threadIdx.x+1];
		}
	}
	__syncthreads();
	
	if(threadIdx.x < 1 && threadIdx.y == 0){
		if(*maxValue1 > 0 && cpy1[0] > 0){
			if(((float)((*maxValue1) - cpy1[0]))/(*maxValue1) < 0.5) {
				(*maxValue1) = 0;
				(*index1) = 0;
			}
		} 
	}
	if(threadIdx.x < 1 && threadIdx.y == 1){
		if(*maxValue2 > 0 && cpy2[0] > 0){
			if(((float)((*maxValue2) - cpy2[0]))/(*maxValue2) < 0.5) {
				(*maxValue2) = 0;
				(*index2) = 0;
			}
		} 
	}
	__syncthreads();
}

/*
__device__ void findMaxCpy(int smem[64], int* maxValue, int* index){
	
	__shared__ unsigned char indexs[32];
	__shared__ int cpy[32];
	if(threadIdx.x < 32 && threadIdx.y == 0){
		if(smem[threadIdx.x+32] > smem[threadIdx.x]){
			cpy[threadIdx.x] = smem[threadIdx.x+32];
			indexs[threadIdx.x] = threadIdx.x+32;
		}else{
			cpy[threadIdx.x] = smem[threadIdx.x];
			indexs[threadIdx.x] = threadIdx.x;
		}
	}
	__syncthreads();
	if(threadIdx.x < 16 && threadIdx.y == 0){
		if(cpy[threadIdx.x+16] > cpy[threadIdx.x]){
			cpy[threadIdx.x] = cpy[threadIdx.x+16];
			indexs[threadIdx.x] = indexs[threadIdx.x+16];
		}
	}
	__syncthreads();
	if(threadIdx.x < 8 && threadIdx.y == 0){
		if(cpy[threadIdx.x+8] > cpy[threadIdx.x]){
			cpy[threadIdx.x] = cpy[threadIdx.x+8];
			indexs[threadIdx.x] = indexs[threadIdx.x+8];
		}
	}
	__syncthreads();
	if(threadIdx.x < 4 && threadIdx.y == 0){
		if(cpy[threadIdx.x+4] > cpy[threadIdx.x]){
			cpy[threadIdx.x] = cpy[threadIdx.x+4];
			indexs[threadIdx.x] = indexs[threadIdx.x+4];
		}
	}
	__syncthreads();
	if(threadIdx.x < 2 && threadIdx.y == 0){
		if(cpy[threadIdx.x+2] > cpy[threadIdx.x]){
			cpy[threadIdx.x] = cpy[threadIdx.x+2];
			indexs[threadIdx.x] = indexs[threadIdx.x+2];
		}
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y == 0){
		if(cpy[threadIdx.x+1] > cpy[threadIdx.x]){
			cpy[threadIdx.x] = cpy[threadIdx.x+1];
			indexs[threadIdx.x] = indexs[threadIdx.x+1];
		}
	}
	__syncthreads();
	
	if(threadIdx.x < 1 && threadIdx.y == 0){
		(*maxValue) = cpy[0];
		(*index) = indexs[0];
		smem[indexs[0]] = 0;
	}
	__syncthreads();

}*/




__device__ void findOrigin(unsigned char s[16][16],  unsigned char pattern[18][18], int* found){
	__shared__ unsigned char patternOriginY[16];
	if(threadIdx.x == 1){
		patternOriginY[threadIdx.y] = 255;
		for(int x = 0; x < 16; x++){
			if(s[x][threadIdx.y] == HORI || s[x][threadIdx.y] == VERT || s[x][threadIdx.y] == DIAG)	{
				patternOriginY[threadIdx.y] = x;
				break;
			}
		}
	}
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0){
		for(int y = 0; y < 16; y++){
			if(patternOriginY[y]!= 255){
				pattern[patternOriginY[y]+1][y+1] = s[patternOriginY[y]][y];
				s[patternOriginY[y]][y] = 0;
				(*found) = 1;
				break;
			}
		} 
	}
	__syncthreads();
}


__device__ void findPatternSize(unsigned char s[18][18], int* size){
	__shared__ int sum[16][16];
	
	if(threadIdx.x < 16 && threadIdx.y < 16){
		sum[threadIdx.x][threadIdx.y] = 0;
	}
	__syncthreads();
	if(threadIdx.x < 16){
		if(s[threadIdx.x+1][threadIdx.y+1] != 0) 
			sum[threadIdx.x][threadIdx.y] = 1;
	}
	__syncthreads();
	if(threadIdx.x < 8 && threadIdx.y < 16){
		sum[threadIdx.x][threadIdx.y]+=sum[threadIdx.x+8][threadIdx.y];
	}
	__syncthreads();
	if(threadIdx.x < 4 && threadIdx.y < 16){
		sum[threadIdx.x][threadIdx.y]+=sum[threadIdx.x+4][threadIdx.y];
	}
	__syncthreads();
	if(threadIdx.x < 2 && threadIdx.y < 16){
		sum[threadIdx.x][threadIdx.y]+=sum[threadIdx.x+2][threadIdx.y];
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y < 16){
		sum[threadIdx.x][threadIdx.y]+=sum[threadIdx.x+1][threadIdx.y];
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y < 8){
		sum[threadIdx.x][threadIdx.y]+=sum[threadIdx.x][threadIdx.y+8];
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y < 4){
		sum[threadIdx.x][threadIdx.y]+=sum[threadIdx.x][threadIdx.y+4];
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y < 2){
		sum[threadIdx.x][threadIdx.y]+=sum[threadIdx.x][threadIdx.y+2];
	}
	__syncthreads();
	if(threadIdx.x < 1 && threadIdx.y < 1){
		sum[threadIdx.x][threadIdx.y]+=sum[threadIdx.x][threadIdx.y+1];
		*size = sum[0][0];
	}
	__syncthreads();
	
}

__device__ void findPattern(unsigned char s[16][16],  unsigned char pattern[18][18], int* size){
	__shared__ int flagi[1];
	flagi[0] = -1;
	if(threadIdx.x < 18){
		pattern[threadIdx.x][threadIdx.y+1] = 0;
		if(threadIdx.y == 0)
			pattern[threadIdx.x][threadIdx.y] = 0;
		if(threadIdx.y == 15)
			pattern[threadIdx.x][threadIdx.y+2] = 0;
	}
	__syncthreads();
	findOrigin( s,  pattern, flagi);
	int r = 0;
	bool b = false;
	__syncthreads();
	//if(flagi[0] == 1){
		do{
			if(threadIdx.x < 16){
				b = false;

				if(s[threadIdx.x][threadIdx.y] > 2 && s[threadIdx.x][threadIdx.y] != NODE){
				 	if(		pattern[threadIdx.x  ][threadIdx.y  ] ||
				 			pattern[threadIdx.x+1][threadIdx.y  ] ||
				 			pattern[threadIdx.x+2][threadIdx.y  ] ||
				 			pattern[threadIdx.x  ][threadIdx.y+1] || 
				 			pattern[threadIdx.x+2][threadIdx.y+1] || 
				 			pattern[threadIdx.x  ][threadIdx.y+2] ||
				 			pattern[threadIdx.x+1][threadIdx.y+2] ||
				 			pattern[threadIdx.x+2][threadIdx.y+2]) {
				 				b = true;
				 				}
				}
			}
			__syncthreads();
			if(threadIdx.x < 16){
				if(b){
					pattern[threadIdx.x+1][threadIdx.y+1] = s[threadIdx.x][threadIdx.y];
					s[threadIdx.x][threadIdx.y] = 0;
				}
			}
			__syncthreads();
		
		}while(r++ <= 25 );
		///__syncthreads();
	//}
	
	findPatternSize(pattern, size);
}

/*__device__ void sumHori(int sums[48][8], int sum[64]){

	/*if(threadIdx.y < 8){
		sums[threadIdx.x][threadIdx.y] += sums[threadIdx.x][threadIdx.y+8];
	}
	__syncthreads();
	if(threadIdx.y < 4){
		sums[threadIdx.x][threadIdx.y] += sums[threadIdx.x][threadIdx.y+4];
	}
	__syncthreads();
	if(threadIdx.y < 2){
		sums[threadIdx.x][threadIdx.y] += sums[threadIdx.x][threadIdx.y+2];
	}
	__syncthreads();
	if(threadIdx.y < 1){
		sums[threadIdx.x][threadIdx.y] += sums[threadIdx.x][threadIdx.y+1];
		sum[threadIdx.x] = sums[threadIdx.x][0]; 
	}
	__syncthreads();	
}*/

__device__ void sumHoriFast(int sums[48][16]){

	if(threadIdx.y < 8){
		sums[threadIdx.x][threadIdx.y] += sums[threadIdx.x][threadIdx.y+8];
	}
	__syncthreads();
	if(threadIdx.y < 4){
		sums[threadIdx.x][threadIdx.y] += sums[threadIdx.x][threadIdx.y+4];
	}
	__syncthreads();
	if(threadIdx.y < 2){
		sums[threadIdx.x][threadIdx.y] += sums[threadIdx.x][threadIdx.y+2];
	}
	__syncthreads();
	if(threadIdx.y < 1){
		sums[threadIdx.x][threadIdx.y] += sums[threadIdx.x][threadIdx.y+1];
	}
	__syncthreads();	
}
__device__ void sumHoriFastDouble(int sums1[48][16], int sums2[48][16]){

	if(threadIdx.y < 8)
		sums1[threadIdx.x][threadIdx.y] += sums1[threadIdx.x][threadIdx.y+8];
	if(threadIdx.y >= 8 && threadIdx.y < 16)
		sums2[threadIdx.x][threadIdx.y-8] += sums2[threadIdx.x][threadIdx.y];
	__syncthreads();
	
	if(threadIdx.y < 4)
		sums1[threadIdx.x][threadIdx.y] += sums1[threadIdx.x][threadIdx.y+4];
	if(threadIdx.y >= 4 && threadIdx.y < 8)
		sums2[threadIdx.x][threadIdx.y-4] += sums2[threadIdx.x][threadIdx.y];
	__syncthreads();
	
	if(threadIdx.y < 2)
		sums1[threadIdx.x][threadIdx.y] += sums1[threadIdx.x][threadIdx.y+2];
	if(threadIdx.y >= 2 && threadIdx.y < 4)
		sums2[threadIdx.x][threadIdx.y-2] += sums2[threadIdx.x][threadIdx.y];
	__syncthreads();
	
	if(threadIdx.y == 0)
		sums1[threadIdx.x][threadIdx.y] += sums1[threadIdx.x][threadIdx.y+1];
	if(threadIdx.y == 1 )
		sums2[threadIdx.x][threadIdx.y-1] += sums2[threadIdx.x][threadIdx.y];
	__syncthreads();	
}

__device__ void comp(unsigned char search[64][16], unsigned char s[16][16], int sums[48][16]){
		if(threadIdx.x > 0 && threadIdx.x < 47){
			int i = 0;
			for(i = 0; i < 16; i++){
			
				if(s[i][threadIdx.y] > 2){
					if(s[i][threadIdx.y] == search[threadIdx.x+i][threadIdx.y] || s[i][threadIdx.y] == search[threadIdx.x+i+1][threadIdx.y] || s[i][threadIdx.y] == search[threadIdx.x+i-1][threadIdx.y] || s[i][threadIdx.y] == NODE){
						if(s[i][threadIdx.y] == VERT || s[i][threadIdx.y] == DIAG ) sums[threadIdx.x][threadIdx.y] += 8;
						else sums[threadIdx.x][threadIdx.y] += 6;
					}
					else if(search[threadIdx.x+i][threadIdx.y] < 3){
						sums[threadIdx.x][threadIdx.y] -=2; 
					}
				}else{
					if( search[threadIdx.x+i][threadIdx.y] > 2)
						 sums[threadIdx.x][threadIdx.y] -=2; 
						 
				}	
			}
		}
}

__device__ void compPattern(unsigned char search[64][16], unsigned char s[18][18], int sums[48][16]){
		if(threadIdx.x > 0 && threadIdx.x < 47){
			int i = 0;
			for(i = 0; i < 16; i++){
				if(s[i+1][threadIdx.y+1] > 2){
					if(s[i+1][threadIdx.y+1] == search[threadIdx.x+i][threadIdx.y] || s[i+1][threadIdx.y+1] == search[threadIdx.x+i+1][threadIdx.y] || s[i+1][threadIdx.y+1] == search[threadIdx.x+i-1][threadIdx.y] || search[i][threadIdx.y] == NODE){
						if(s[i+1][threadIdx.y+1] == VERT ) sums[threadIdx.x][threadIdx.y] += 5;
						else if( s[i+1][threadIdx.y+1] == DIAG)  sums[threadIdx.x][threadIdx.y] += 7;
						else sums[threadIdx.x][threadIdx.y] += 4;
					}
					else if(search[threadIdx.x+i][threadIdx.y] < 3){
						sums[threadIdx.x][threadIdx.y] -=2; 
					}
				}	
			}
		}
}
#define MIN_PAT_SIZE 10
#define DIFF_TH_OCL 30
#define DIFF_TH_FLAT 2
#define DIFF_TH_FLAT_SOBEL 20
#define DIFF_TH_OCL_SOBEL 10
#define MIN_SUM 10


__global__ void edgeMacher(const int rows, const int cols, unsigned char *left, unsigned char *right, unsigned char *srcL, unsigned char *srcR, unsigned char *disp){

	__shared__ unsigned char s[16][16];
	__shared__ unsigned char l[18][18];
	__shared__ unsigned char pattern1[18][18];
	__shared__ unsigned char pattern2[18][18];
	__shared__ unsigned char r[48+16+2][18];
	__shared__ unsigned char search[48+16][16];
	__shared__ int sums1[48][16];
	__shared__ int sums2[48][16];
	__shared__ unsigned char flag[1];
	__shared__ unsigned char cpy1[24];
	__shared__ unsigned char cpy2[24];
	
	__shared__ int results1[3];
	__shared__ int results2[3];
	
	int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;
	
	sums1[threadIdx.x][threadIdx.y] = 0;
	sums2[threadIdx.x][threadIdx.y] = 0;
	
	if(threadIdx.x < 18){
		pattern1[threadIdx.x][threadIdx.y+1] = 0;
		if(threadIdx.y == 0)
			pattern1[threadIdx.x][threadIdx.y] = 0;
		if(threadIdx.y == 15)
			pattern1[threadIdx.x][threadIdx.y+2] = 0;
	}	
	if(threadIdx.x < 18){
		pattern2[threadIdx.x][threadIdx.y+1] = 0;
		if(threadIdx.y == 0)
			pattern2[threadIdx.x][threadIdx.y] = 0;
		if(threadIdx.y == 15)
			pattern2[threadIdx.x][threadIdx.y+2] = 0;
	}
	
	
    if(x < 0 || x >= cols - 64 || y < 0 || y >= rows) return;
    
	int iX = threadIdx.x+1;
	int iY = threadIdx.y+1;
    
    search[threadIdx.x][threadIdx.y] = right[y*cols+x];
	r[iX][iY] = srcR[y*cols+x];
    if(threadIdx.x < 16){
		s[threadIdx.x][threadIdx.y] = left[y*cols+x];
		if(s[threadIdx.x][threadIdx.y] > 2) disp[y*cols+x] = 255;
		l[iX][iY] = srcL[y*cols+x];
		search[threadIdx.x+48][threadIdx.y] = right[y*cols+(x+48)];
		
		if(iX < 2){
			if(x < 1){ 
				l[iX-1][iY] = 0;
				if(iY < 2) l[iX-1][iY-1] = 0;
			}else{
				l[iX-1][iY] = srcL[y*cols+x-1];
				if(y < 1) l[iX-1][iY-1] = 0;
				else l[iX-1][iY-1] = srcL[(y-1)*cols+(x-1)];  		
			}
		}
		if(iX >= 16){
			if(x >= cols-1){ 
				l[iX+1][iY] = 0;
				if(iY >= blockDim.y) l[iX+1][iY+1] = 0;
			}else{
				l[iX+1][iY] = srcL[y*cols+(x+1)];
				if(y >= rows-1) l[iX+1][iY+1] = 0;
				else l[iX+1][iY+1] = srcL[(y+1)*cols+x+1];
			}
		}
		if(iY < 2){
			if(y < 1){ 
				l[iX][iY-1] = 0;
				if(iX >= 16) l[iX+1][iY-1] = 0;
			}else{
				l[iX][iY-1] = srcL[(y-1)*cols+x];
				if(x >= cols-1) l[iX+1][iY-1] = 0;
				else l[iX+1][iY-1] = srcL[(y-1)*cols+x+1];
			}
		}
		if(iY >= blockDim.y){
			if(y >= rows-1){ 
				l[iX][iY+1] = 0;
				if(iX < 2) l[iX-1][iY+1] = 0;
			}else{
				l[iX][iY+1] = srcL[(y+1)*cols+x];
				if(x < 1) l[iX-1][iY+1] = 0;
				else l[iX-1][iY+1] = srcL[(y+1)*cols+(x-1)];
			}
		}
	}
	
	if(iX < 2){
    	if(x < 1){ 
    		r[iX-1][iY] = 0;
			if(iY < 2) r[iX-1][iY-1] = 0;
    	}else{
    		r[iX-1][iY] = srcR[y*cols+x-1];
    		if(y < 1) r[iX-1][iY-1] = 0;
    		else r[iX-1][iY-1] = srcR[(y-1)*cols+(x-1)];  		
    	}
    }
    if(iX >= blockDim.x-17){
    	if(x >= cols-1){ 
    		r[iX+17][iY] = 0;
			if(iY >= blockDim.y) r[iX+1][iY+1] = 0;
    	}else{
    		r[iX+17][iY] = srcR[y*cols+(x+17)];
    		if(y >= rows-1) r[iX+17][iY+1] = 0;
    		else r[iX+17][iY+1] = srcR[(y+1)*cols+x+17];
    	}
    }
    if(iY < 2){
    	if(y < 1){ 
    		r[iX][iY-1] = 0;
			if(iX >= blockDim.x) r[iX+1][iY-1] = 0;
    	}else{
    		r[iX][iY-1] = srcR[(y-1)*cols+x];
    		if(x >= cols-1) r[iX+17][iY-1] = 0;
    		else r[iX+17][iY-1] = srcR[(y-1)*cols+x+17];
    	}
    }
    if(iY >= blockDim.y){
    	if(y >= rows-1){ 
    		r[iX][iY+1] = 0;
			if(iX < 2) r[iX-1][iY+1] = 0;
    	}else{
    		r[iX][iY+1] = srcR[(y+1)*cols+x];
    		if(x < 1) r[iX-1][iY+1] = 0;
    		else r[iX-1][iY+1] = srcR[(y+1)*cols+(x-1)];
    	}
    }

	if(threadIdx.x == 0 && threadIdx.y == 0)
		flag[0] = 0;
	do{
		sums1[threadIdx.x][threadIdx.y] = 0;
		sums2[threadIdx.x][threadIdx.y] = 0;
		if(threadIdx.y == 1){
			if(threadIdx.x < 3){
					results1[threadIdx.x] = 0;
					results2[threadIdx.x] = 0;
				}
		}

		__syncthreads();
	
		findPattern(s,  pattern1, &results1[0]);
		
		__syncthreads();
		if(results1[0] < MIN_PAT_SIZE)
			findPattern(s,  pattern1, &results1[0]);
		
		__syncthreads();
		//if(results1[0] >= MIN_PAT_SIZE)
		findPattern(s,  pattern2, &results2[0]);
		__syncthreads();
		if(results2[0] < MIN_PAT_SIZE)
			findPattern(s,  pattern2, &results2[0]);
			
		__syncthreads();
		if(results1[0] >= MIN_PAT_SIZE)
			compPattern(search, pattern1, sums1);
		if(results2[0] >= MIN_PAT_SIZE)
			compPattern(search, pattern2, sums2);
			
		//comp(search, s, sums);	
		
		__syncthreads();
		if(results1[0] >= MIN_PAT_SIZE || results2[0] >= MIN_PAT_SIZE)
			sumHoriFastDouble(sums1,  sums2);
		//sumHoriFast( sums1);
	
		__syncthreads();

		//findMaxFast(sums, indexs, &results[1], &results[0]);
		if(results1[0] >= MIN_PAT_SIZE || results2[0] >= MIN_PAT_SIZE)
			//findMaxFastDouble(sums1, sums2, &results1[1], &results1[2], &results2[1], &results2[2]);
			findMaxFastDoubleCpy(sums1, sums2, &results1[1], &results1[2], &results2[1], &results2[2], cpy1, cpy2);
		__syncthreads();
		
		if(!(x < 1 || x >= cols - 64-1 || y < 1 || y >= rows-1)){ 
			if(threadIdx.x < 16){
				if(results1[0] >= MIN_PAT_SIZE){
					if(pattern1[threadIdx.x+1][threadIdx.y+1] == HORI){
						int diff1 = (int)l[iX][iY-1] - r[iX+results1[1]][iY-1];
						int diff2 = (int)l[iX][iY+1] - r[iX+results1[1]][iY+1];
						int sumL = 0;
						int sumR = 0;
						sumL-=  l[iX-1][iY-1];
    					sumL-=2*l[iX  ][iY-1];
    					sumL-=  l[iX+1][iY-1];
    					sumL+=  l[iX-1][iY+1];
    					sumL+=2*l[iX  ][iY+1];
    					sumL+=  l[iX+1][iY+1];
    			
						sumR-=  r[iX-1+results1[1]][iY-1];
    					sumR-=2*r[iX+results1[1]][iY-1];
    					sumR-=  r[iX+1+results1[1]][iY-1];
    					sumR+=  r[iX-1+results1[1]][iY+1];
    					sumR+=2*r[iX+results1[1]][iY+1];
    					sumR+=  r[iX+1+results1[1]][iY+1];
    					
    					int diff3 = sumL-sumR;
						bool upperOcl = (diff1 < DIFF_TH_OCL && diff1 > -DIFF_TH_OCL); 
						bool lowerOcl = (diff2 < DIFF_TH_OCL && diff2 > -DIFF_TH_OCL); 
						bool sobelOcl = (diff3 < DIFF_TH_OCL_SOBEL && diff3 > -DIFF_TH_OCL_SOBEL);
						bool upperFlat = (diff1 < DIFF_TH_FLAT && diff1 > -DIFF_TH_FLAT); 
						bool lowerFlat = (diff2 < DIFF_TH_FLAT && diff2 > -DIFF_TH_FLAT); 
						bool sobelFlat = (diff3 < DIFF_TH_FLAT_SOBEL && diff3 > -DIFF_TH_FLAT_SOBEL);
						if(!sobelOcl && (upperOcl || lowerOcl) && results1[2] >= MIN_SUM){
							disp[y*cols+x] = results1[1]*4;
						}
						if(((upperFlat || lowerFlat) && sobelFlat) && results1[2] >= MIN_SUM){
							disp[y*cols+x] = results1[1]*4;
						}
					}else if(pattern1[threadIdx.x+1][threadIdx.y+1] == VERT){
						int diff1 = (int)l[iX-1][iY] - r[iX+results1[1]-1][iY];
						int diff2 = (int)l[iX+1][iY] - r[iX+results1[1]+1][iY];
						int sumL = 0;
						int sumR = 0;
						sumL-=  l[iX-1][iY-1];
    					sumL-=2*l[iX-1][iY  ];
    					sumL-=  l[iX-1][iY+1];
    					sumL+=  l[iX+1][iY-1];
    					sumL+=2*l[iX+1][iY  ];
    					sumL+=  l[iX+1][iY+1];
    			
						sumR-=  r[iX-1+results1[1]][iY-1];
    					sumR-=2*r[iX-1+results1[1]][iY  ];
    					sumR-=  r[iX-1+results1[1]][iY+1];
    					sumR+=  r[iX+1+results1[1]][iY-1];
    					sumR+=2*r[iX+1+results1[1]][iY  ];
    					sumR+=  r[iX+1+results1[1]][iY+1];
    					int diff3 = sumL-sumR;
						bool leftOcl = (diff1 < DIFF_TH_OCL && diff1 > -DIFF_TH_OCL); 
						bool rightOcl = (diff2 < DIFF_TH_OCL && diff2 > -DIFF_TH_OCL); 
						bool sobelOcl = (diff3 < DIFF_TH_OCL_SOBEL && diff3 > -DIFF_TH_OCL_SOBEL);
						bool leftFlat = (diff1 < DIFF_TH_FLAT && diff1 > -DIFF_TH_FLAT); 
						bool rightFlat = (diff2 < DIFF_TH_FLAT && diff2 > -DIFF_TH_FLAT);  
						bool sobelFlat = (diff3 < DIFF_TH_FLAT_SOBEL && diff3 > -DIFF_TH_FLAT_SOBEL);
						if(!sobelOcl && (leftOcl || rightOcl) && results1[2] >= MIN_SUM){
							disp[y*cols+x] = results1[1]*4;
						}
						if(((leftFlat || rightFlat) && sobelFlat) && results1[2] >= MIN_SUM){
							disp[y*cols+x] = results1[1]*4;
						}
					}/*else if(results1[2] >= MIN_SUM && pattern1[threadIdx.x+1][threadIdx.y+1]){
						disp[y*cols+x] = results1[1]*4;
					}*/
				}
			}
			if(threadIdx.x < 16){
				if(results2[0] >= MIN_PAT_SIZE){
					if(pattern2[threadIdx.x+1][threadIdx.y+1] == HORI){
						int diff1 = (int)l[iX][iY-1] - r[iX+results2[1]][iY-1];
						int diff2 = (int)l[iX][iY+1] - r[iX+results2[1]][iY+1];
						int sumL = 0;
						int sumR = 0;
						sumL-=  l[iX-1][iY-1];
    					sumL-=2*l[iX  ][iY-1];
    					sumL-=  l[iX+1][iY-1];
    					sumL+=  l[iX-1][iY+1];
    					sumL+=2*l[iX  ][iY+1];
    					sumL+=  l[iX+1][iY+1];
    			
						sumR-=  r[iX-1+results1[1]][iY-1];
    					sumR-=2*r[iX+results1[1]][iY-1];
    					sumR-=  r[iX+1+results1[1]][iY-1];
    					sumR+=  r[iX-1+results1[1]][iY+1];
    					sumR+=2*r[iX+results1[1]][iY+1];
    					sumR+=  r[iX+1+results1[1]][iY+1];
    					
    					int diff3 = sumL-sumR;
						bool upperOcl = (diff1 < DIFF_TH_OCL && diff1 > -DIFF_TH_OCL); 
						bool lowerOcl = (diff2 < DIFF_TH_OCL && diff2 > -DIFF_TH_OCL); 
						bool sobelOcl = (diff3 < DIFF_TH_OCL_SOBEL && diff3 > -DIFF_TH_OCL_SOBEL);
						bool upperFlat = (diff1 < DIFF_TH_FLAT && diff1 > -DIFF_TH_FLAT); 
						bool lowerFlat = (diff2 < DIFF_TH_FLAT && diff2 > -DIFF_TH_FLAT); 
						bool sobelFlat = (diff3 < DIFF_TH_FLAT_SOBEL && diff3 > -DIFF_TH_FLAT_SOBEL);
						if(!sobelOcl && (upperOcl || lowerOcl) && results2[2] >= MIN_SUM){
							disp[y*cols+x] = results2[1]*4;
						}
						if(((upperFlat || lowerFlat) && sobelFlat) && results2[2] >= MIN_SUM){
							disp[y*cols+x] = results2[1]*4;
						}
					}else if(pattern2[threadIdx.x+1][threadIdx.y+1] == VERT){
						int diff1 = (int)l[iX-1][iY] - r[iX+results2[1]-1][iY];
						int diff2 = (int)l[iX+1][iY] - r[iX+results2[1]+1][iY];
						int sumL = 0;
						int sumR = 0;
						sumL-=  l[iX-1][iY-1];
    					sumL-=2*l[iX-1][iY  ];
    					sumL-=  l[iX-1][iY+1];
    					sumL+=  l[iX+1][iY-1];
    					sumL+=2*l[iX+1][iY  ];
    					sumL+=  l[iX+1][iY+1];
    			
						sumR-=  r[iX-1+results1[1]][iY-1];
    					sumR-=2*r[iX-1+results1[1]][iY  ];
    					sumR-=  r[iX-1+results1[1]][iY+1];
    					sumR+=  r[iX+1+results1[1]][iY-1];
    					sumR+=2*r[iX+1+results1[1]][iY  ];
    					sumR+=  r[iX+1+results1[1]][iY+1];
    					int diff3 = sumL-sumR;
						bool leftOcl = (diff1 < DIFF_TH_OCL && diff1 > -DIFF_TH_OCL); 
						bool rightOcl = (diff2 < DIFF_TH_OCL && diff2 > -DIFF_TH_OCL); 
						bool sobelOcl = (diff3 < DIFF_TH_OCL_SOBEL && diff3 > -DIFF_TH_OCL_SOBEL);
						bool leftFlat = (diff1 < DIFF_TH_FLAT && diff1 > -DIFF_TH_FLAT); 
						bool rightFlat = (diff2 < DIFF_TH_FLAT && diff2 > -DIFF_TH_FLAT);  
						bool sobelFlat = (diff3 < DIFF_TH_FLAT_SOBEL && diff3 > -DIFF_TH_FLAT_SOBEL);
						if(!sobelOcl && (leftOcl || rightOcl) && results2[2] >= MIN_SUM){
							disp[y*cols+x] = results2[1]*4;
						}
						if(((leftFlat || rightFlat) && sobelFlat) && results2[2] >= MIN_SUM){
							disp[y*cols+x] = results2[1]*4;
						}
					}/*else if(results1[2] >= MIN_SUM && pattern1[threadIdx.x+1][threadIdx.y+1]){
						disp[y*cols+x] = results1[1]*4;
					}*/
				}
			}
	}
		if(threadIdx.x == 0 && threadIdx.y == 0)
			flag[0]++;	
		__syncthreads();
	}while(flag[0] < 1);
	
		/*if(threadIdx.y == 0){
				if(sum[threadIdx.x] > 15){
					if(threadIdx.x == results[0])
						disp[(blockIdx.y * 16 + (int)(threadIdx.x/16))*cols+blockIdx.x * 16 + (threadIdx.x-(16*(int)(threadIdx.x/16)))] = 255;
					else 
						disp[(blockIdx.y * 16 + (int)(threadIdx.x/16))*cols+blockIdx.x * 16 + (threadIdx.x-(16*(int)(threadIdx.x/16)))] = 128;
				}
				
		}*/
}



__global__ void edgeEstimate(const int rows, const int cols, unsigned char *left, unsigned char *leftLow, unsigned char *disp, unsigned char *dispOut,const int mode ){
	    
	    __shared__ unsigned char dis[26][26];
	    __shared__ unsigned char disTemp[24][24];
	    __shared__ unsigned char edge[26][26];
	    __shared__ unsigned char flags[1];
	    
	    
    	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

		int iX = threadIdx.x+1;
		int iY = threadIdx.y+1;
    	
    	if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	cpyToShared24x24xR1(disp, dis, &x, &y, rows, cols);
    	cpyToShared24x24xR1(left, edge, &x, &y, rows, cols);
    	
    	if(threadIdx.x == 5 && threadIdx.y == 5)
    		flags[0] = 0;
    	
    	__syncthreads();
    	disTemp[threadIdx.x][threadIdx.y] = 0;
   		do{   			
			__syncthreads();
			if(edge[iX][iY] >= HORI || edge[iX][iY] <= NODE_U  ){
				if(dis[iX][iY] == 255){
					if(!(edge[iX-1][iY-1] == NODE ||
	   						edge[iX][iY-1] ==  NODE ||
	   						edge[iX+1][iY-1] ==  NODE ||
	   						edge[iX+1][iY] ==  NODE ||
	   						edge[iX-1][iY] ==  NODE ||
	   						edge[iX-1][iY+1] ==  NODE ||
	   						edge[iX][iY+1] ==  NODE ||
	   						edge[iX+1][iY+1] ==  NODE)){
	   					
						if(dis[iX-1][iY-1] != 0 && dis[iX-1][iY-1] < 254)
							disTemp[threadIdx.x][threadIdx.y] = dis[iX-1][iY-1];
						else if(dis[iX][iY-1] != 0 && dis[iX][iY-1] < 254)
							disTemp[threadIdx.x][threadIdx.y] = dis[iX][iY-1];
						else if(dis[iX+1][iY-1] != 0 && dis[iX+1][iY-1] < 254)
							disTemp[threadIdx.x][threadIdx.y] = dis[iX+1][iY-1];
						else if(dis[iX-1][iY] != 0 && dis[iX-1][iY] < 254)
							disTemp[threadIdx.x][threadIdx.y] = dis[iX-1][iY];
						else if(dis[iX+1][iY] != 0 && dis[iX+1][iY] < 254)
							disTemp[threadIdx.x][threadIdx.y] = dis[iX+1][iY];
						else if(dis[iX-1][iY+1] != 0 && dis[iX-1][iY+1] < 254)
							disTemp[threadIdx.x][threadIdx.y] = dis[iX-1][iY+1];
						else if(dis[iX][iY+1] != 0 && dis[iX][iY+1] < 254)
							disTemp[threadIdx.x][threadIdx.y] = dis[iX][iY+1];
						else if(dis[iX+1][iY+1] != 0 && dis[iX+1][iY+1] < 254)
							disTemp[threadIdx.x][threadIdx.y] = dis[iX+1][iY+1];
					}
				}
			}
			/*if(mode == 0){
				if(edgeLow[iX][iY] == 0 && dis[iX][iY] == 0){
					if(	!((edge[iX][iY-1] >= HORI && edge[iX][iY-1]  <= NODE_U) ||
						  (edge[iX-1][iY] >= HORI && edge[iX-1][iY]  <= NODE_U) ||
						  (edge[iX][iY+1] >= HORI && edge[iX][iY+1]  <= NODE_U) ||
						  (edge[iX+1][iY] >= HORI && edge[iX+1][iY]  <= NODE_U))){
						  if((dis[iX][iY-1] == 254) ||
						  	 (dis[iX-1][iY] == 254) ||
						   	 (dis[iX][iY+1] == 254) ||
						  	 (dis[iX+1][iY] == 254)){
								disTemp[threadIdx.x][threadIdx.y] = 254;
		   				}
		   			}
				}
				if(edgeLow[iX][iY] == 0 && dis[iX][iY] == 0 && disTemp[threadIdx.x][threadIdx.y] == 0){
					if(	!((edge[iX][iY-1] >= HORI && edge[iX][iY-1]  <= NODE_U) ||
						  (edge[iX-1][iY] >= HORI && edge[iX-1][iY]  <= NODE_U) ||
						  (edge[iX][iY+1] >= HORI && edge[iX][iY+1]  <= NODE_U) ||
						  (edge[iX+1][iY] >= HORI && edge[iX+1][iY]  <= NODE_U))){
						  if((dis[iX][iY-1] == 253) ||
						  	 (dis[iX-1][iY] == 253) ||
						   	 (dis[iX][iY+1] == 253) ||
						  	 (dis[iX+1][iY] == 253)){
								disTemp[threadIdx.x][threadIdx.y] = 253;
		   				}
		   			}
				}
			__syncthreads();
			}
			if(mode == 1){
				if(dis[iX][iY] == 254 ){	
					if (dis[iX-1][iY-1] < 254 && dis[iX-1][iY-1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX-1][iY-1];
					if (dis[iX  ][iY-1] < 254 && dis[iX  ][iY-1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX  ][iY-1];
					if (dis[iX+1][iY-1] < 254 && dis[iX+1][iY-1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX+1][iY-1];
					if (dis[iX+1][iY  ] < 254 && dis[iX+1][iY  ] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX+1][iY  ];
					if (dis[iX+1][iY+1] < 254 && dis[iX+1][iY+1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX+1][iY+1];
					if (dis[iX  ][iY+1] < 254 && dis[iX  ][iY+1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX  ][iY+1];
					if (dis[iX-1][iY+1] < 254 && dis[iX-1][iY+1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX-1][iY+1];
					if (dis[iX-1][iY  ] < 254 && dis[iX-1][iY  ] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX-1][iY  ];
				}
			
				if(dis[iX][iY] == 253 && disTemp[threadIdx.x][threadIdx.y] == 0){	
					if (dis[iX-1][iY-1] < 253 && dis[iX-1][iY-1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX-1][iY-1];
					if (dis[iX  ][iY-1] < 253 && dis[iX  ][iY-1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX  ][iY-1];
					if (dis[iX+1][iY-1] < 253 && dis[iX+1][iY-1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX+1][iY-1];
					if (dis[iX+1][iY  ] < 253 && dis[iX+1][iY  ] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX+1][iY  ];
					if (dis[iX+1][iY+1] < 253 && dis[iX+1][iY+1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX+1][iY+1];
					if (dis[iX  ][iY+1] < 253 && dis[iX  ][iY+1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX  ][iY+1];
					if (dis[iX-1][iY+1] < 253 && dis[iX-1][iY+1] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX-1][iY+1];
					if (dis[iX-1][iY  ] < 253 && dis[iX-1][iY  ] > 0 )  disTemp[threadIdx.x][threadIdx.y] = dis[iX-1][iY  ];
				}
			__syncthreads();
			}*/
			if(disTemp[threadIdx.x][threadIdx.y] != 0){
   				dis[iX][iY] = disTemp[threadIdx.x][threadIdx.y];
   				disTemp[threadIdx.x][threadIdx.y] = 0;
   			}
   			
			if(threadIdx.x == 5 && threadIdx.y == 5)
				flags[0] += 1;
			__syncthreads();
		    	
		}while(flags[0] < 30);
		//__syncthreads();
		
		//if(edge[iX][iY] == NODE)
		//	dispOut[y*cols+x] = 100;
		//else if(edge[threadIdx.x][threadIdx.y] == NODE)
			//dispOut[y*cols+x] = 255;
		//else
		
		dispOut[y*cols+x] = dis[iX][iY];  
				
		//dispOut[y*cols+x] = dis[iX][iY];
}


__device__ void match2x64p(unsigned char pattern[2][2], unsigned char search[64+2][2], unsigned int result[64][2]){

	result[threadIdx.x][threadIdx.y]  = abs(((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y])); 
	result[threadIdx.x][threadIdx.y] += abs(((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y])); 

}

__device__ void sum2x64(unsigned int result[64][2]){
	if(threadIdx.y == 0)
		result[threadIdx.x][threadIdx.y] += result[threadIdx.x][threadIdx.y+1];
}

__global__ void edgeMatch2(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY ) {
	
	__shared__  unsigned char pattern[2][2];
	__shared__  unsigned char search[64+2][2];
	__shared__ unsigned int result[64][2];
	
	
	int x = blockIdx.x * 2 + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int t = (y+shiftY)*cols+(x+shiftX);
	
    
    if(x < 0 || x >= cols-64-2-shiftX || y < 0 || y >= rows-shiftY) return;
    
    search[threadIdx.x][threadIdx.y] = edgeR[t];
    result[threadIdx.x][threadIdx.y] = 0;
    if(threadIdx.x < 2){
    	pattern[threadIdx.x][threadIdx.y] = edgeL[t];
    	search[threadIdx.x+64][threadIdx.y] = edgeR[t+64];
    }
    __syncthreads();
    
    match2x64p(pattern, search, result);
    __syncthreads();
    
    sum2x64(result);
    __syncthreads();
    
	if(threadIdx.y == 0)
		out[(blockIdx.y*(320)+blockIdx.x)*64+threadIdx.x] = result[threadIdx.x][0];
	
}


__device__ void matchEdge4x64(unsigned char pattern[4][4], unsigned char search[64+4][4], int result[64][4]){
	
	result[threadIdx.x][threadIdx.y] = pattern[0][threadIdx.y] == search[threadIdx.x  ][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[0][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[1][threadIdx.y] == search[threadIdx.x+1][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[1][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[2][threadIdx.y] == search[threadIdx.x+2][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[2][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[3][threadIdx.y] == search[threadIdx.x+3][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[3][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
}

__device__ void match4x64(unsigned char pattern[4][4], unsigned char search[64+4][4], unsigned int result[64][4]){

	result[threadIdx.x][threadIdx.y]  = abs((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += 4*abs((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += 4*abs((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += abs((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) ; 
}

__device__ void match4x64p(unsigned char pattern[4][4], unsigned char search[64+4][4], unsigned int result[64][4]){

	result[threadIdx.x][threadIdx.y]  = 2*((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) * ((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += 4*((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]) * ((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]); 
	result[threadIdx.x][threadIdx.y] += 4*((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) * ((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += 2*((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) * ((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) ;	
}

__device__ void sum4x64(unsigned int result[64][4]){
	if(threadIdx.y < 2)
		result[threadIdx.x][threadIdx.y] += result[threadIdx.x][threadIdx.y+2];
	__syncthreads();
	if(threadIdx.y == 0)
		result[threadIdx.x][threadIdx.y] += result[threadIdx.x][threadIdx.y+1];
	
}

__global__ void edgeMatch4(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY ) {
	
	__shared__  unsigned char pattern[4][4];
	__shared__  unsigned char search[64+4][4];
	__shared__ unsigned int result[64][4];
	
	
	int x = blockIdx.x * 4 + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int t = (y+shiftY)*cols+(x+shiftX);
	
    
    if(x < 0 || x >= cols-64-4-shiftX || y < 0 || y >= rows-shiftY) return;
    
    search[threadIdx.x][threadIdx.y] = edgeR[t];
    result[threadIdx.x][threadIdx.y] = 0;
    if(threadIdx.x < 4){
    	pattern[threadIdx.x][threadIdx.y] = edgeL[t];
    	search[threadIdx.x+64][threadIdx.y] = edgeR[t+64];
    }
    __syncthreads();
    
    match4x64p(pattern, search, result);
    __syncthreads();
    
    sum4x64(result);
    __syncthreads();
    

	if(threadIdx.y == 0)
		out[(blockIdx.y*(160)+blockIdx.x)*64+threadIdx.x] = result[threadIdx.x][0];
	
}

__device__ void matchEdge8x64(unsigned char pattern[8][8], unsigned char search[64+8][8], int result[64][8]){
	
	result[threadIdx.x][threadIdx.y] = pattern[0][threadIdx.y] == search[threadIdx.x  ][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[0][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[1][threadIdx.y] == search[threadIdx.x+1][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[1][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[2][threadIdx.y] == search[threadIdx.x+2][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[2][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[3][threadIdx.y] == search[threadIdx.x+3][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[3][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[4][threadIdx.y] == search[threadIdx.x+4][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[4][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[5][threadIdx.y] == search[threadIdx.x+5][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[5][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[6][threadIdx.y] == search[threadIdx.x+6][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[6][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
	result[threadIdx.x][threadIdx.y] = pattern[7][threadIdx.y] == search[threadIdx.x+7][threadIdx.y] ? ++result[threadIdx.x][threadIdx.y] : 
		(pattern[7][threadIdx.y] ? --result[threadIdx.x][threadIdx.y] : result[threadIdx.x][threadIdx.y]);
}

__device__ void match8x64(unsigned char pattern[8][8], unsigned char search[64+8][8], unsigned int result[64][8]){
	
	result[threadIdx.x][threadIdx.y]  = abs((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += abs((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += abs((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += abs((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) ;	
	result[threadIdx.x][threadIdx.y] += abs((int)pattern[4][threadIdx.y] - search[threadIdx.x+4][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += abs((int)pattern[5][threadIdx.y] - search[threadIdx.x+5][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += abs((int)pattern[6][threadIdx.y] - search[threadIdx.x+7][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += abs((int)pattern[7][threadIdx.y] - search[threadIdx.x+8][threadIdx.y]) ;
}
__device__ void match8x64p(unsigned char pattern[8][8], unsigned char search[64+8][8], unsigned int result[64][8]){
	
	result[threadIdx.x][threadIdx.y]  =   ((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) * ((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += 2*((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]) * ((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]); 
	result[threadIdx.x][threadIdx.y] += 4*((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) * ((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += 4*((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) * ((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) ;	
	result[threadIdx.x][threadIdx.y] += 4*((int)pattern[4][threadIdx.y] - search[threadIdx.x+4][threadIdx.y]) * ((int)pattern[4][threadIdx.y] - search[threadIdx.x+4][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += 4*((int)pattern[5][threadIdx.y] - search[threadIdx.x+5][threadIdx.y]) * ((int)pattern[5][threadIdx.y] - search[threadIdx.x+5][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] += 2*((int)pattern[6][threadIdx.y] - search[threadIdx.x+7][threadIdx.y]) * ((int)pattern[6][threadIdx.y] - search[threadIdx.x+7][threadIdx.y]) ; 
	result[threadIdx.x][threadIdx.y] +=   ((int)pattern[7][threadIdx.y] - search[threadIdx.x+8][threadIdx.y]) * ((int)pattern[7][threadIdx.y] - search[threadIdx.x+8][threadIdx.y]);
}

__device__ void sum8x64(unsigned int result[64][8]){
	if(threadIdx.y < 4)
		result[threadIdx.x][threadIdx.y] += result[threadIdx.x][threadIdx.y+4];
	__syncthreads();
	if(threadIdx.y < 2)
		result[threadIdx.x][threadIdx.y] += result[threadIdx.x][threadIdx.y+2];
	__syncthreads();
	if(threadIdx.y == 0)
		result[threadIdx.x][threadIdx.y] += result[threadIdx.x][threadIdx.y+1];
}

__global__ void edgeMatch8(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY ) {
	
	__shared__  unsigned char pattern[8][8];
	__shared__  unsigned char search[64+8][8];
	__shared__ unsigned int result[64][8];
	
	
	int x = blockIdx.x * 8 + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int t = (y+shiftY)*cols+x+shiftX;
	
    
    if(x < 0 || x >= cols-64-8-shiftX || y < 0 || y >= rows-shiftY) return;
    
    search[threadIdx.x][threadIdx.y] = edgeR[t];
    result[threadIdx.x][threadIdx.y] = 0;
    if(threadIdx.x < 8){
    	pattern[threadIdx.x][threadIdx.y] = edgeL[t];
    	search[threadIdx.x+64][threadIdx.y] = edgeR[t+64];
    }
    __syncthreads();
    
    match8x64(pattern, search, result);
    __syncthreads();
    
    sum8x64(result);
    __syncthreads();
    
	if(threadIdx.y == 0)
		out[(blockIdx.y*(80)+blockIdx.x)*64+threadIdx.x] = result[threadIdx.x][0];
	
}

__device__ void matchEdge12x64(unsigned char pattern[12][12], unsigned char search[64+12][12], unsigned int* result){
	result[threadIdx.x+threadIdx.y*64] = pattern[0][threadIdx.y] == search[threadIdx.x  ][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[0][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[1][threadIdx.y] == search[threadIdx.x+1][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[1][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[2][threadIdx.y] == search[threadIdx.x+2][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[2][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[3][threadIdx.y] == search[threadIdx.x+3][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[3][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);	
	result[threadIdx.x+threadIdx.y*64] = pattern[4][threadIdx.y] == search[threadIdx.x+4][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[4][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[5][threadIdx.y] == search[threadIdx.x+5][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[5][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[6][threadIdx.y] == search[threadIdx.x+6][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[6][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[7][threadIdx.y] == search[threadIdx.x+7][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[7][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[8][threadIdx.y] == search[threadIdx.x+8][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[8][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[9][threadIdx.y] == search[threadIdx.x+9][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[9][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[10][threadIdx.y] == search[threadIdx.x+10][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[10][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
	result[threadIdx.x+threadIdx.y*64] = pattern[11][threadIdx.y] == search[threadIdx.x+11][threadIdx.y] ? ++result[threadIdx.x+threadIdx.y*64] : 
		(pattern[11][threadIdx.y] ? --result[threadIdx.x+threadIdx.y*64] : result[threadIdx.x+threadIdx.y*64]);
		
}

__device__ void match12x64(unsigned char pattern[12][12], unsigned char search[64+12][12], unsigned int* result){
	result[threadIdx.y*64+threadIdx.x]  = abs((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*abs((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 3*abs((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*abs((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += 4*abs((int)pattern[4][threadIdx.y] - search[threadIdx.x+4][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*abs((int)pattern[5][threadIdx.y] - search[threadIdx.x+5][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*abs((int)pattern[6][threadIdx.y] - search[threadIdx.x+6][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*abs((int)pattern[7][threadIdx.y] - search[threadIdx.x+7][threadIdx.y]) ;
	result[threadIdx.y*64+threadIdx.x] += 4*abs((int)pattern[8][threadIdx.y] - search[threadIdx.x+8][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 3*abs((int)pattern[9][threadIdx.y] - search[threadIdx.x+9][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*abs((int)pattern[10][threadIdx.y] - search[threadIdx.x+10][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[11][threadIdx.y] - search[threadIdx.x+11][threadIdx.y]) ;	
}

__device__ void sum12x64(unsigned int *result){
	if(threadIdx.y < 4)
		result[threadIdx.x+threadIdx.y*64] = result[threadIdx.x+(threadIdx.y+8)*64];
	__syncthreads();
	if(threadIdx.y < 4)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+4)*64];
	__syncthreads();
	if(threadIdx.y < 2)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+2)*64];
	__syncthreads();
	if(threadIdx.y == 0)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+1)*64];
	
}


/*__global__ void edgeMatch12(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY ) {
	
	__shared__  unsigned char pattern[12][12];
	__shared__  unsigned char search[64+12][12];
	extern __shared__ unsigned int result[];
	
	
	int x = blockIdx.x * 12 + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int t = (y+shiftY)*cols+(x+shiftX);
	
    
    if(x < 0 || x >= cols-64-12-shiftX || y < 0 || y >= rows-shiftY) return;
    
    search[threadIdx.x][threadIdx.y] = edgeR[t];
    result[threadIdx.x+threadIdx.y*64] = 0;
    if(threadIdx.x < 12){
    	pattern[threadIdx.x][threadIdx.y] = edgeL[t];
    	search[threadIdx.x+64][threadIdx.y] = edgeR[t+64];
    }
    __syncthreads();
    
    match12x64(pattern, search, result);
    __syncthreads();
    
    sum12x64(result);
    __syncthreads();
    

	if(threadIdx.y == 0)
		out[(blockIdx.y*(54)+blockIdx.x)*64+threadIdx.x] = result[threadIdx.x];
	
}*/

__device__ void matchEdge16x64(unsigned char pattern[16][16], unsigned char search[64+16][16], int* result){
	
	int index = threadIdx.x+threadIdx.y*64;
	result[index] = pattern[0][threadIdx.y] == search[threadIdx.x  ][threadIdx.y] ? ++result[index] : 
		(pattern[0][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[1][threadIdx.y] == search[threadIdx.x+1][threadIdx.y] ? ++result[index] : 
		(pattern[1][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[2][threadIdx.y] == search[threadIdx.x+2][threadIdx.y] ? ++result[index] : 
		(pattern[2][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[3][threadIdx.y] == search[threadIdx.x+3][threadIdx.y] ? ++result[index] : 
		(pattern[3][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[4][threadIdx.y] == search[threadIdx.x+4][threadIdx.y] ? ++result[index] : 
		(pattern[4][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[5][threadIdx.y] == search[threadIdx.x+5][threadIdx.y] ? ++result[index] : 
		(pattern[5][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[6][threadIdx.y] == search[threadIdx.x+6][threadIdx.y] ? ++result[index] : 
		(pattern[6][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[7][threadIdx.y] == search[threadIdx.x+7][threadIdx.y] ? ++result[index] : 
		(pattern[7][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[8][threadIdx.y] == search[threadIdx.x+8][threadIdx.y] ? ++result[index] : 
		(pattern[8][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[9][threadIdx.y] == search[threadIdx.x+9][threadIdx.y] ? ++result[index] : 
		(pattern[9][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[10][threadIdx.y] == search[threadIdx.x+10][threadIdx.y] ? ++result[index] : 
		(pattern[10][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[11][threadIdx.y] == search[threadIdx.x+11][threadIdx.y] ? ++result[index] : 
		(pattern[11][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[12][threadIdx.y] == search[threadIdx.x+12][threadIdx.y] ? ++result[index] : 
		(pattern[12][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[13][threadIdx.y] == search[threadIdx.x+13][threadIdx.y] ? ++result[index] : 
		(pattern[13][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[14][threadIdx.y] == search[threadIdx.x+14][threadIdx.y] ? ++result[index] : 
		(pattern[14][threadIdx.y] ? --result[index] : result[index]);
	result[index] = pattern[15][threadIdx.y] == search[threadIdx.x+15][threadIdx.y] ? ++result[index] : 
		(pattern[15][threadIdx.y] ? --result[index] : result[index]);
	
}

__device__ void match16x64(unsigned char pattern[16][16], unsigned char search[64+16][16], unsigned int* result){
	result[threadIdx.y*64+threadIdx.x]  = abs((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[4][threadIdx.y] - search[threadIdx.x+4][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[5][threadIdx.y] - search[threadIdx.x+5][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[6][threadIdx.y] - search[threadIdx.x+6][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[7][threadIdx.y] - search[threadIdx.x+7][threadIdx.y]) ;
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[8][threadIdx.y] - search[threadIdx.x+8][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[9][threadIdx.y] - search[threadIdx.x+9][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[10][threadIdx.y] - search[threadIdx.x+10][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[11][threadIdx.y] - search[threadIdx.x+11][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[12][threadIdx.y] - search[threadIdx.x+12][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[13][threadIdx.y] - search[threadIdx.x+13][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[14][threadIdx.y] - search[threadIdx.x+14][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs((int)pattern[15][threadIdx.y] - search[threadIdx.x+15][threadIdx.y]) ;	
}

__device__ void match16x64p(unsigned char pattern[16][16], unsigned char search[64+16][16], unsigned int* result){
	result[threadIdx.y*64+threadIdx.x]  = ((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) * ((int)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]) *((int)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 3*((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) * ((int)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) *((int)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[4][threadIdx.y] - search[threadIdx.x+4][threadIdx.y]) * ((int)pattern[4][threadIdx.y] - search[threadIdx.x+4][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[5][threadIdx.y] - search[threadIdx.x+5][threadIdx.y]) * ((int)pattern[5][threadIdx.y] - search[threadIdx.x+5][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[6][threadIdx.y] - search[threadIdx.x+6][threadIdx.y]) * ((int)pattern[6][threadIdx.y] - search[threadIdx.x+6][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[7][threadIdx.y] - search[threadIdx.x+7][threadIdx.y]) * ((int)pattern[7][threadIdx.y] - search[threadIdx.x+7][threadIdx.y]) ;
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[8][threadIdx.y] - search[threadIdx.x+8][threadIdx.y]) *((int)pattern[8][threadIdx.y] - search[threadIdx.x+8][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[9][threadIdx.y] - search[threadIdx.x+9][threadIdx.y]) *((int)pattern[9][threadIdx.y] - search[threadIdx.x+9][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[10][threadIdx.y] - search[threadIdx.x+10][threadIdx.y])*((int)pattern[10][threadIdx.y] - search[threadIdx.x+10][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[11][threadIdx.y] - search[threadIdx.x+11][threadIdx.y]) *((int)pattern[11][threadIdx.y] - search[threadIdx.x+11][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += 4*((int)pattern[12][threadIdx.y] - search[threadIdx.x+12][threadIdx.y]) * ((int)pattern[12][threadIdx.y] - search[threadIdx.x+12][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 3*((int)pattern[13][threadIdx.y] - search[threadIdx.x+13][threadIdx.y]) *((int)pattern[13][threadIdx.y] - search[threadIdx.x+13][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*((int)pattern[14][threadIdx.y] - search[threadIdx.x+14][threadIdx.y]) * ((int)pattern[14][threadIdx.y] - search[threadIdx.x+14][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += ((int)pattern[15][threadIdx.y] - search[threadIdx.x+15][threadIdx.y]) * ((int)pattern[15][threadIdx.y] - search[threadIdx.x+15][threadIdx.y]) ;	
}

__device__ void sum16x64(unsigned int *result){
	if(threadIdx.y < 8)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+8)*64];
	__syncthreads();
	if(threadIdx.y < 4)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+4)*64];
	__syncthreads();
	if(threadIdx.y < 2)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+2)*64];
	__syncthreads();
	if(threadIdx.y == 0)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+1)*64];
}


__global__ void edgeMatch16(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY ) {
	
	__shared__  unsigned char pattern[16][16];
	__shared__  unsigned char search[64+16][16];
	extern __shared__ unsigned int result[];
	
	
	int x = blockIdx.x * 16 + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int t = (y+shiftY)*cols+x+shiftX;
	
    
    if(x < 0 || x >= cols-64-16-shiftX || y < 0 || y >= rows-shiftY) return;
    
    search[threadIdx.x][threadIdx.y] = edgeR[t];
    result[threadIdx.y*64+threadIdx.x] = 0;
    if(threadIdx.x < 16){
    	pattern[threadIdx.x][threadIdx.y] = edgeL[t];
    	search[threadIdx.x+64][threadIdx.y] = edgeR[t+64];
    }
    __syncthreads();
    
    match16x64(pattern, search, result);
    __syncthreads();
    
    sum16x64(result);
    __syncthreads();
    

	if(threadIdx.y == 0)
		out[(blockIdx.y*(40)+blockIdx.x)*64+threadIdx.x] = result[threadIdx.x];
}

__device__ void match32x64f(unsigned char pattern[32][32], unsigned char search[64+32][32], unsigned int* result){
	result[threadIdx.y*64+threadIdx.x]  = (((int)pattern[0][threadIdx.y]) - search[threadIdx.x  ][threadIdx.y])*(((int)pattern[0][threadIdx.y]) - search[threadIdx.x  ][threadIdx.y])  ; 
	result[threadIdx.y*64+threadIdx.x] += (((int)pattern[1][threadIdx.y]) - search[threadIdx.x+1][threadIdx.y])*(((int)pattern[1][threadIdx.y]) - search[threadIdx.x+1][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += (((int)pattern[2][threadIdx.y]) - search[threadIdx.x+2][threadIdx.y]) *abs(((int)pattern[2][threadIdx.y]) - search[threadIdx.x+2][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += (((int)pattern[3][threadIdx.y]) - search[threadIdx.x+3][threadIdx.y]) *abs(((int)pattern[3][threadIdx.y]) - search[threadIdx.x+3][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += 2*(((int)pattern[4][threadIdx.y]) - search[threadIdx.x+4][threadIdx.y]) *abs(((int)pattern[4][threadIdx.y]) - search[threadIdx.x+4][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*(((int)pattern[5][threadIdx.y]) - search[threadIdx.x+5][threadIdx.y])*(((int)pattern[5][threadIdx.y]) - search[threadIdx.x+5][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*(((int)pattern[6][threadIdx.y]) - search[threadIdx.x+6][threadIdx.y]) *(((int)pattern[6][threadIdx.y]) - search[threadIdx.x+6][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*(((int)pattern[7][threadIdx.y]) - search[threadIdx.x+7][threadIdx.y]) *(((int)pattern[7][threadIdx.y]) - search[threadIdx.x+7][threadIdx.y]) ;
	result[threadIdx.y*64+threadIdx.x] += 3*(((int)pattern[8][threadIdx.y]) - search[threadIdx.x+8][threadIdx.y]) *(((int)pattern[8][threadIdx.y]) - search[threadIdx.x+8][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 3*(((int)pattern[9][threadIdx.y]) - search[threadIdx.x+9][threadIdx.y]) *(((int)pattern[9][threadIdx.y]) - search[threadIdx.x+9][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 3*(((int)pattern[10][threadIdx.y]) - search[threadIdx.x+10][threadIdx.y])*(((int)pattern[10][threadIdx.y]) - search[threadIdx.x+10][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 3*(((int)pattern[11][threadIdx.y]) - search[threadIdx.x+11][threadIdx.y]) *(((int)pattern[11][threadIdx.y]) - search[threadIdx.x+11][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += 3*(((int)pattern[12][threadIdx.y]) - search[threadIdx.x+12][threadIdx.y])*(((int)pattern[12][threadIdx.y]) - search[threadIdx.x+12][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*(((int)pattern[13][threadIdx.y]) - search[threadIdx.x+13][threadIdx.y])*(((int)pattern[13][threadIdx.y]) - search[threadIdx.x+13][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*(((int)pattern[14][threadIdx.y]) - search[threadIdx.x+14][threadIdx.y])*(((int)pattern[14][threadIdx.y]) - search[threadIdx.x+14][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*(((int)pattern[15][threadIdx.y]) - search[threadIdx.x+15][threadIdx.y])*(((int)pattern[15][threadIdx.y]) - search[threadIdx.x+15][threadIdx.y]) ;
	result[threadIdx.y*64+threadIdx.x] += 4*(((int)pattern[16][threadIdx.y]) - search[threadIdx.x+16][threadIdx.y])*(((int)pattern[16][threadIdx.y]) - search[threadIdx.x+16][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*(((int)pattern[17][threadIdx.y]) - search[threadIdx.x+17][threadIdx.y])*(((int)pattern[17][threadIdx.y]) - search[threadIdx.x+17][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*(((int)pattern[18][threadIdx.y]) - search[threadIdx.x+18][threadIdx.y])*(((int)pattern[18][threadIdx.y]) - search[threadIdx.x+18][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 4*(((int)pattern[19][threadIdx.y] )- search[threadIdx.x+19][threadIdx.y])*(((int)pattern[19][threadIdx.y] )- search[threadIdx.x+19][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += 3*(((int)pattern[20][threadIdx.y]) - search[threadIdx.x+20][threadIdx.y])*(((int)pattern[20][threadIdx.y]) - search[threadIdx.x+20][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 3*(((int)pattern[21][threadIdx.y])- search[threadIdx.x+21][threadIdx.y])*(((int)pattern[21][threadIdx.y])- search[threadIdx.x+21][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 3*(((int)pattern[22][threadIdx.y]) - search[threadIdx.x+22][threadIdx.y]) *(((int)pattern[22][threadIdx.y]) - search[threadIdx.x+22][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += 3*(((int)pattern[23][threadIdx.y] )- search[threadIdx.x+23][threadIdx.y])*(((int)pattern[23][threadIdx.y] )- search[threadIdx.x+23][threadIdx.y]) ;
	result[threadIdx.y*64+threadIdx.x] += 2*(((int)pattern[24][threadIdx.y]) - search[threadIdx.x+24][threadIdx.y])*(((int)pattern[24][threadIdx.y]) - search[threadIdx.x+24][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*(((int)pattern[25][threadIdx.y]) - search[threadIdx.x+25][threadIdx.y])*(((int)pattern[25][threadIdx.y]) - search[threadIdx.x+25][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*(((int)pattern[26][threadIdx.y]) - search[threadIdx.x+26][threadIdx.y])*(((int)pattern[26][threadIdx.y]) - search[threadIdx.x+26][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += 2*(((int)pattern[27][threadIdx.y] )- search[threadIdx.x+27][threadIdx.y])*(((int)pattern[27][threadIdx.y] )- search[threadIdx.x+27][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += (((int)pattern[28][threadIdx.y]) - search[threadIdx.x+28][threadIdx.y])*(((int)pattern[28][threadIdx.y]) - search[threadIdx.x+28][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += (((int)pattern[29][threadIdx.y]) - search[threadIdx.x+29][threadIdx.y])*(((int)pattern[29][threadIdx.y]) - search[threadIdx.x+29][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += (((int)pattern[30][threadIdx.y] )- search[threadIdx.x+30][threadIdx.y])*(((int)pattern[30][threadIdx.y] )- search[threadIdx.x+30][threadIdx.y])  ; 
	result[threadIdx.y*64+threadIdx.x] += (((int)pattern[31][threadIdx.y]) - search[threadIdx.x+31][threadIdx.y])*(((int)pattern[31][threadIdx.y]) - search[threadIdx.x+31][threadIdx.y]);	
}

__device__ void match32x64s(unsigned char pattern[32][32], unsigned char search[64+32][32], unsigned int* result){
	result[(threadIdx.y+16)*64+threadIdx.x]  = (((int)pattern[0][threadIdx.y+16]) - search[threadIdx.x  ][threadIdx.y+16])*(((int)pattern[0][threadIdx.y+16]) - search[threadIdx.x  ][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += (((int)pattern[1][threadIdx.y+16]) - search[threadIdx.x+1][threadIdx.y+16])*(((int)pattern[1][threadIdx.y+16]) - search[threadIdx.x+1][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += (((int)pattern[2][threadIdx.y+16]) - search[threadIdx.x+2][threadIdx.y+16])*(((int)pattern[2][threadIdx.y+16]) - search[threadIdx.x+2][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += (((int)pattern[3][threadIdx.y+16]) - search[threadIdx.x+3][threadIdx.y+16])*(((int)pattern[3][threadIdx.y+16]) - search[threadIdx.x+3][threadIdx.y+16]) ;	
	result[(threadIdx.y+16)*64+threadIdx.x] += 2*(((int)pattern[4][threadIdx.y+16]) - search[threadIdx.x+4][threadIdx.y+16])*(((int)pattern[4][threadIdx.y+16]) - search[threadIdx.x+4][threadIdx.y+16])  ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 2*(((int)pattern[5][threadIdx.y+16]) - search[threadIdx.x+5][threadIdx.y+16])*(((int)pattern[5][threadIdx.y+16]) - search[threadIdx.x+5][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 2*(((int)pattern[6][threadIdx.y+16]) - search[threadIdx.x+6][threadIdx.y+16])*(((int)pattern[6][threadIdx.y+16]) - search[threadIdx.x+6][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 2*(((int)pattern[7][threadIdx.y+16]) - search[threadIdx.x+7][threadIdx.y+16])*(((int)pattern[7][threadIdx.y+16]) - search[threadIdx.x+7][threadIdx.y+16]) ;
	result[(threadIdx.y+16)*64+threadIdx.x] += 3*(((int)pattern[8][threadIdx.y+16]) - search[threadIdx.x+8][threadIdx.y+16]) *(((int)pattern[8][threadIdx.y+16]) - search[threadIdx.x+8][threadIdx.y+16])  ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 3*(((int)pattern[9][threadIdx.y+16]) - search[threadIdx.x+9][threadIdx.y+16])*(((int)pattern[9][threadIdx.y+16]) - search[threadIdx.x+9][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 3*(((int)pattern[10][threadIdx.y+16]) - search[threadIdx.x+10][threadIdx.y+16])*(((int)pattern[10][threadIdx.y+16]) - search[threadIdx.x+10][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 3*(((int)pattern[11][threadIdx.y+16]) - search[threadIdx.x+11][threadIdx.y+16])*(((int)pattern[11][threadIdx.y+16]) - search[threadIdx.x+11][threadIdx.y+16]) ;	
	result[(threadIdx.y+16)*64+threadIdx.x] += 4*(((int)pattern[12][threadIdx.y+16]) - search[threadIdx.x+12][threadIdx.y+16])*(((int)pattern[12][threadIdx.y+16]) - search[threadIdx.x+12][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 4*(((int)pattern[13][threadIdx.y+16]) - search[threadIdx.x+13][threadIdx.y+16])*(((int)pattern[13][threadIdx.y+16]) - search[threadIdx.x+13][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 4*(((int)pattern[14][threadIdx.y+16]) - search[threadIdx.x+14][threadIdx.y+16])*(((int)pattern[14][threadIdx.y+16]) - search[threadIdx.x+14][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 4*(((int)pattern[15][threadIdx.y+16]) - search[threadIdx.x+15][threadIdx.y+16])*(((int)pattern[15][threadIdx.y+16]) - search[threadIdx.x+15][threadIdx.y+16]) ;
	result[(threadIdx.y+16)*64+threadIdx.x] += 4*(((int)pattern[16][threadIdx.y+16]) - search[threadIdx.x+16][threadIdx.y+16])*(((int)pattern[16][threadIdx.y+16]) - search[threadIdx.x+16][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += 4*(((int)pattern[17][threadIdx.y+16]) - search[threadIdx.x+17][threadIdx.y+16])*(((int)pattern[17][threadIdx.y+16]) - search[threadIdx.x+17][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 4*(((int)pattern[18][threadIdx.y+16]) - search[threadIdx.x+18][threadIdx.y+16])*(((int)pattern[18][threadIdx.y+16]) - search[threadIdx.x+18][threadIdx.y+16])  ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 4*(((int)pattern[19][threadIdx.y+16] )- search[threadIdx.x+19][threadIdx.y+16])*(((int)pattern[19][threadIdx.y+16] )- search[threadIdx.x+19][threadIdx.y+16]) ;	
	result[(threadIdx.y+16)*64+threadIdx.x] += 3*(((int)pattern[20][threadIdx.y+16]) - search[threadIdx.x+20][threadIdx.y+16])*(((int)pattern[20][threadIdx.y+16]) - search[threadIdx.x+20][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 3*(((int)pattern[21][threadIdx.y+16])- search[threadIdx.x+21][threadIdx.y+16])*(((int)pattern[21][threadIdx.y+16])- search[threadIdx.x+21][threadIdx.y+16])  ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 3*(((int)pattern[22][threadIdx.y+16]) - search[threadIdx.x+22][threadIdx.y+16])*(((int)pattern[22][threadIdx.y+16]) - search[threadIdx.x+22][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 3*(((int)pattern[23][threadIdx.y+16] )- search[threadIdx.x+23][threadIdx.y+16])*(((int)pattern[23][threadIdx.y+16] )- search[threadIdx.x+23][threadIdx.y+16]) ;
	result[(threadIdx.y+16)*64+threadIdx.x] += 2*(((int)pattern[24][threadIdx.y+16]) - search[threadIdx.x+24][threadIdx.y+16])*(((int)pattern[24][threadIdx.y+16]) - search[threadIdx.x+24][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 2*(((int)pattern[25][threadIdx.y+16]) - search[threadIdx.x+25][threadIdx.y+16])*(((int)pattern[25][threadIdx.y+16]) - search[threadIdx.x+25][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 2*(((int)pattern[26][threadIdx.y+16]) - search[threadIdx.x+26][threadIdx.y+16])*(((int)pattern[26][threadIdx.y+16]) - search[threadIdx.x+26][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += 2*(((int)pattern[27][threadIdx.y+16] )- search[threadIdx.x+27][threadIdx.y+16]) *(((int)pattern[27][threadIdx.y+16] )- search[threadIdx.x+27][threadIdx.y+16]);	
	result[(threadIdx.y+16)*64+threadIdx.x] += (((int)pattern[28][threadIdx.y+16]) - search[threadIdx.x+28][threadIdx.y+16]) *(((int)pattern[28][threadIdx.y+16]) - search[threadIdx.x+28][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += (((int)pattern[29][threadIdx.y+16]) - search[threadIdx.x+29][threadIdx.y+16])*(((int)pattern[29][threadIdx.y+16]) - search[threadIdx.x+29][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += (((int)pattern[30][threadIdx.y+16] )- search[threadIdx.x+30][threadIdx.y+16])*(((int)pattern[30][threadIdx.y+16] )- search[threadIdx.x+30][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += (((int)pattern[31][threadIdx.y+16]) - search[threadIdx.x+31][threadIdx.y+16])*(((int)pattern[31][threadIdx.y+16]) - search[threadIdx.x+31][threadIdx.y+16]) ;	
}

__device__ void match32x64fp(unsigned char pattern[32][32], unsigned char search[64+32][32], unsigned int* result){
	result[threadIdx.y*64+threadIdx.x]  = abs(((int)pattern[0][threadIdx.y]) - search[threadIdx.x  ][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[1][threadIdx.y]) - search[threadIdx.x+1][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[2][threadIdx.y]) - search[threadIdx.x+2][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[3][threadIdx.y]) - search[threadIdx.x+3][threadIdx.y])  ;	
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[4][threadIdx.y]) - search[threadIdx.x+4][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[5][threadIdx.y]) - search[threadIdx.x+5][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[6][threadIdx.y]) - search[threadIdx.x+6][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[7][threadIdx.y]) - search[threadIdx.x+7][threadIdx.y]);
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[8][threadIdx.y]) - search[threadIdx.x+8][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[9][threadIdx.y]) - search[threadIdx.x+9][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[10][threadIdx.y]) - search[threadIdx.x+10][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[11][threadIdx.y]) - search[threadIdx.x+11][threadIdx.y]);	
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[12][threadIdx.y]) - search[threadIdx.x+12][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[13][threadIdx.y]) - search[threadIdx.x+13][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[14][threadIdx.y]) - search[threadIdx.x+14][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[15][threadIdx.y]) - search[threadIdx.x+15][threadIdx.y]) ;
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[16][threadIdx.y]) - search[threadIdx.x+16][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[17][threadIdx.y]) - search[threadIdx.x+17][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[18][threadIdx.y]) - search[threadIdx.x+18][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[19][threadIdx.y] )- search[threadIdx.x+19][threadIdx.y]) ;	
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[20][threadIdx.y]) - search[threadIdx.x+20][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[21][threadIdx.y])- search[threadIdx.x+21][threadIdx.y]) ; 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[22][threadIdx.y]) - search[threadIdx.x+22][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[23][threadIdx.y] )- search[threadIdx.x+23][threadIdx.y]) ;
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[24][threadIdx.y]) - search[threadIdx.x+24][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[25][threadIdx.y]) - search[threadIdx.x+25][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[26][threadIdx.y]) - search[threadIdx.x+26][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[27][threadIdx.y] )- search[threadIdx.x+27][threadIdx.y]);	
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[28][threadIdx.y]) - search[threadIdx.x+28][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[29][threadIdx.y]) - search[threadIdx.x+29][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[30][threadIdx.y] )- search[threadIdx.x+30][threadIdx.y]); 
	result[threadIdx.y*64+threadIdx.x] += abs(((int)pattern[31][threadIdx.y]) - search[threadIdx.x+31][threadIdx.y]);	
}

__device__ void match32x64sp(unsigned char pattern[32][32], unsigned char search[64+32][32], unsigned int* result){
	result[(threadIdx.y+16)*64+threadIdx.x]  = abs(((int)pattern[0][threadIdx.y+16]) - search[threadIdx.x  ][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[1][threadIdx.y+16]) - search[threadIdx.x+1][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[2][threadIdx.y+16]) - search[threadIdx.x+2][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[3][threadIdx.y+16]) - search[threadIdx.x+3][threadIdx.y+16]);	
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[4][threadIdx.y+16]) - search[threadIdx.x+4][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[5][threadIdx.y+16]) - search[threadIdx.x+5][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[6][threadIdx.y+16]) - search[threadIdx.x+6][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[7][threadIdx.y+16]) - search[threadIdx.x+7][threadIdx.y+16]);
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[8][threadIdx.y+16]) - search[threadIdx.x+8][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[9][threadIdx.y+16]) - search[threadIdx.x+9][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[10][threadIdx.y+16]) - search[threadIdx.x+10][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[11][threadIdx.y+16]) - search[threadIdx.x+11][threadIdx.y+16]);	
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[12][threadIdx.y+16]) - search[threadIdx.x+12][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[13][threadIdx.y+16]) - search[threadIdx.x+13][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[14][threadIdx.y+16]) - search[threadIdx.x+14][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[15][threadIdx.y+16]) - search[threadIdx.x+15][threadIdx.y+16]);
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[16][threadIdx.y+16]) - search[threadIdx.x+16][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[17][threadIdx.y+16]) - search[threadIdx.x+17][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[18][threadIdx.y+16]) - search[threadIdx.x+18][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[19][threadIdx.y+16] )- search[threadIdx.x+19][threadIdx.y+16]);	
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[20][threadIdx.y+16]) - search[threadIdx.x+20][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[21][threadIdx.y+16])- search[threadIdx.x+21][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[22][threadIdx.y+16]) - search[threadIdx.x+22][threadIdx.y+16]) ; 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[23][threadIdx.y+16] )- search[threadIdx.x+23][threadIdx.y+16]);
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[24][threadIdx.y+16]) - search[threadIdx.x+24][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[25][threadIdx.y+16]) - search[threadIdx.x+25][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[26][threadIdx.y+16]) - search[threadIdx.x+26][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[27][threadIdx.y+16] )- search[threadIdx.x+27][threadIdx.y+16]);	
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[28][threadIdx.y+16]) - search[threadIdx.x+28][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[29][threadIdx.y+16]) - search[threadIdx.x+29][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[30][threadIdx.y+16] )- search[threadIdx.x+30][threadIdx.y+16]); 
	result[(threadIdx.y+16)*64+threadIdx.x] += abs(((int)pattern[31][threadIdx.y+16]) - search[threadIdx.x+31][threadIdx.y+16]);	
}


__device__ void sum32x64(unsigned int *result){
	if(threadIdx.y < 16)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+16)*64];
	__syncthreads();
	if(threadIdx.y < 8)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+8)*64];
	__syncthreads();
	if(threadIdx.y < 4)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+4)*64];
	__syncthreads();
	if(threadIdx.y < 2)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+2)*64];
	__syncthreads();
	if(threadIdx.y == 0)
		result[threadIdx.x+threadIdx.y*64] += result[threadIdx.x+(threadIdx.y+1)*64];
}




__global__ void edgeMatch32(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY ) {
	
	__shared__  unsigned char pattern[32][32];
	__shared__  unsigned char search[64+32][32];
	extern __shared__ unsigned int result[];
	
	
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	int t = (y+shiftY)*cols+x+shiftX;
	int t2 = (y+16+shiftY)*cols+x+shiftX;	
    
    if(x < 0 || x >= cols-64-32-shiftX || y < 0 || y >= rows-shiftY) return;
    
    search[threadIdx.x][threadIdx.y] = edgeR[t];
    search[threadIdx.x][threadIdx.y+16] = edgeR[t2];
    result[threadIdx.y*64+threadIdx.x] = 0;
    result[(threadIdx.y+16)*64+threadIdx.x] = 0;
    __syncthreads();
    if(threadIdx.x < 32){
    	pattern[threadIdx.x][threadIdx.y] = edgeL[t];
    	pattern[threadIdx.x][threadIdx.y+16] = edgeL[t2];
    	search[threadIdx.x+64][threadIdx.y] = edgeR[t+64];
    	search[threadIdx.x+64][threadIdx.y+16] = edgeR[t2+64];
    }
    __syncthreads();
    
    match32x64fp(pattern, search, result);
    match32x64sp(pattern, search, result);
    __syncthreads();
    
    sum32x64(result);
    __syncthreads();
    

	if(threadIdx.y == 0)
		out[(blockIdx.y*(20)+blockIdx.x)*64+threadIdx.x] = result[threadIdx.x];
	
}
__device__ unsigned short absus(short a)
{
  return max(-a, a);
}
__device__ void match8x64w16(unsigned char pattern[16][16], unsigned char search[64+16][16], unsigned short* result, unsigned short* idx){
	unsigned short* result2 = (unsigned short*)&result[1024];
	// Left half
	result[*idx]  = absus((short)pattern[0][threadIdx.y] - search[threadIdx.x  ][threadIdx.y]) ; 
	result[*idx] += absus((short)pattern[1][threadIdx.y] - search[threadIdx.x+1][threadIdx.y]) ; 
	result[*idx] += absus((short)pattern[2][threadIdx.y] - search[threadIdx.x+2][threadIdx.y]) ; 
	result[*idx] += absus((short)pattern[3][threadIdx.y] - search[threadIdx.x+3][threadIdx.y]) ;	
	result[*idx] += absus((short)pattern[4][threadIdx.y] - search[threadIdx.x+4][threadIdx.y]) ; 
	result[*idx] += absus((short)pattern[5][threadIdx.y] - search[threadIdx.x+5][threadIdx.y]) ; 
	result[*idx] += absus((short)pattern[6][threadIdx.y] - search[threadIdx.x+6][threadIdx.y]) ; 
	result[*idx] += absus((short)pattern[7][threadIdx.y] - search[threadIdx.x+7][threadIdx.y]) ;
	// Right half
	result2[*idx] =  absus((short)pattern[8][threadIdx.y] - search[threadIdx.x+8][threadIdx.y]) ; 
	result2[*idx] += absus((short)pattern[9][threadIdx.y] - search[threadIdx.x+9][threadIdx.y]) ; 
	result2[*idx] += absus((short)pattern[10][threadIdx.y] - search[threadIdx.x+10][threadIdx.y]) ; 
	result2[*idx] += absus((short)pattern[11][threadIdx.y] - search[threadIdx.x+11][threadIdx.y]) ;	
	result2[*idx] += absus((short)pattern[12][threadIdx.y] - search[threadIdx.x+12][threadIdx.y]) ; 
	result2[*idx] += absus((short)pattern[13][threadIdx.y] - search[threadIdx.x+13][threadIdx.y]) ; 
	result2[*idx] += absus((short)pattern[14][threadIdx.y] - search[threadIdx.x+14][threadIdx.y]) ; 
	result2[*idx] += absus((short)pattern[15][threadIdx.y] - search[threadIdx.x+15][threadIdx.y]) ;	
}

__device__ void sum8x64w16(unsigned short *result, unsigned short* idx){
	unsigned short* result2 = (unsigned short*)&result[1024];
	if(threadIdx.y < 4){
		 result[*idx] +=  result[threadIdx.x+(threadIdx.y+4)*64];
		result2[*idx] += result2[threadIdx.x+(threadIdx.y+4)*64];
	}
	if(threadIdx.y > 7 && threadIdx.y < 12)	{
		 result[*idx] +=  result[threadIdx.x+(threadIdx.y+4)*64];
		result2[*idx] += result2[threadIdx.x+(threadIdx.y+4)*64];
	}
	__syncthreads();
	if(threadIdx.y < 2){
		 result[*idx] +=  result[threadIdx.x+(threadIdx.y+2)*64];
		result2[*idx] += result2[threadIdx.x+(threadIdx.y+2)*64];
		}
	if(threadIdx.y > 7 && threadIdx.y < 10){
		 result[*idx] +=  result[threadIdx.x+(threadIdx.y+2)*64];
		result2[*idx] += result2[threadIdx.x+(threadIdx.y+2)*64];
		}
	__syncthreads();
	
	if(threadIdx.y == 0){
		 result[*idx] +=  result[threadIdx.x+(threadIdx.y+1)*64];
		result2[*idx] += result2[threadIdx.x+(threadIdx.y+1)*64];
		}
	if(threadIdx.y == 8){
		 result[*idx] +=  result[threadIdx.x+(threadIdx.y+1)*64];
		result2[*idx] += result2[threadIdx.x+(threadIdx.y+1)*64];
		}
}


__global__ void edgeMatch8w16(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned short *out) {
	
	__shared__  unsigned char pattern[16][16];
	__shared__  unsigned char search[64+16][16];
	extern __shared__ unsigned short results[];
	
	
	int x = blockIdx.x * 16 + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int t = y*cols+x;
	unsigned short idx = threadIdx.y*64+threadIdx.x;
	
    //result[threadIdx.y*64+threadIdx.x] = 0; possible setting max uchar
    if(x < 0 || x >= cols-64-16 || y < 0 || y >= rows) return;
    
    search[threadIdx.x][threadIdx.y] = edgeR[t];
    results[idx] = 0;
    if(threadIdx.x < 16){
    	pattern[threadIdx.x][threadIdx.y] = edgeL[t];
    	search[threadIdx.x+64][threadIdx.y] = edgeR[t+64];
    }
    __syncthreads();
    
    match8x64w16(pattern, search, results, &idx);
    __syncthreads();
    
    sum8x64w16(results, &idx);
    __syncthreads();
    
	if(threadIdx.y == 0){
		out[(2*blockIdx.y*(80)+2*blockIdx.x)*64+threadIdx.x] = results[idx];
		out[(2*blockIdx.y*(80)+(2*blockIdx.x+1))*64+threadIdx.x] = results[1024+idx];
	}
	if(threadIdx.y == 8){
		out[((2*(blockIdx.y)+1)*(80)+2*blockIdx.x)*64+threadIdx.x] = results[idx];
		out[((2*(blockIdx.y)+1)*(80)+2*blockIdx.x+1)*64+threadIdx.x] = results[1024+idx];
	}
}
/*__device__ void findBestIter(unsigned int kernel[4][4][64], unsigned char out[4][4]){

	int maxVal = 50;
	out[threadIdx.x][threadIdx.y] = 0;
	for(int i = 0; i < 64; i++){
		if(kernel[threadIdx.x][threadIdx.y][i] > maxVal){
			maxVal = kernel[threadIdx.x][threadIdx.y][i];
			out[threadIdx.x][threadIdx.y] = i;
		}	
	}
}

__global__ void brain(const int rows, const int cols, unsigned char *edgeL, int *dis4, int *dis8, int *dis12, int *dis16, unsigned char *disp) {

	__shared__  int kernel[4][4][64];
	__shared__  unsigned char edge[4][4];
	__shared__  unsigned char out[4][4];
	
	int x = blockIdx.x * blockDim.y + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int t = y*cols+x;
	
	out[threadIdx.x][threadIdx.y] = 0;
	edge[threadIdx.x][threadIdx.y] = 0;
	__syncthreads();
	edge[threadIdx.x][threadIdx.y] = edgeL[t];
	__syncthreads();
	if(edge[threadIdx.x][threadIdx.y]){
		for(int i = 0; i < 64; i++){
			kernel[threadIdx.x][threadIdx.y][i] = 16*dis4[(blockIdx.y*(160)+blockIdx.x)*64+i];
			kernel[threadIdx.x][threadIdx.y][i] += 4*dis8[(blockIdx.y/2*(80)+blockIdx.x/2)*64+i];
			kernel[threadIdx.x][threadIdx.y][i] += 3*dis12[(blockIdx.y/3*(54)+blockIdx.x/3)*64+i];
			kernel[threadIdx.x][threadIdx.y][i] += dis16[(blockIdx.y/4*(40)+blockIdx.x/4)*64+i];
		} 
		//__syncthreads();
		findBestIter(kernel, out);
	}
	if(edge[threadIdx.x][threadIdx.y]){
		disp[t] = out[threadIdx.x][threadIdx.y]*4;
	}
}*/

__device__ void findBest64(unsigned int kernel[64], unsigned char out[1], int *th){
	__shared__  unsigned char idx[32];
	out[1] = 0;
	idx[threadIdx.z] = threadIdx.z;
	__syncthreads();
	if(threadIdx.z < 32){
		if(kernel[threadIdx.z+32] > kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+32];
			idx[threadIdx.z] = threadIdx.z+32;
		}else
			idx[threadIdx.z] = threadIdx.z;
	}
	__syncthreads();
	if(threadIdx.z < 16){
		if(kernel[threadIdx.z+16] > kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+16];
			idx[threadIdx.z] = idx[threadIdx.z+16];
		}
	}
	__syncthreads();
	if(threadIdx.z < 8){
		if(kernel[threadIdx.z+8] > kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+8];
			idx[threadIdx.z] = idx[threadIdx.z+8];
		}
	}
	__syncthreads();
	if(threadIdx.z < 4){
		if(kernel[threadIdx.z+4] > kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+4];
			idx[threadIdx.z] = idx[threadIdx.z+4];
		}
	}
	__syncthreads();
	if(threadIdx.z < 2){
		if(kernel[threadIdx.z+2] > kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+2];
			idx[threadIdx.z] = idx[threadIdx.z+2];
		}
	}
	__syncthreads();
	if(threadIdx.z < 1){
		if(kernel[threadIdx.z+1] > kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+1];
			idx[threadIdx.z] = idx[threadIdx.z+1];
		}
		if(kernel[threadIdx.z] < *th && idx[threadIdx.z] < 50)
			out[0] = idx[threadIdx.z];
		else
			out[0] = 0;
	}
	
	
}

__device__ void findMin64(unsigned int kernel[64], unsigned char out[1], int *th){
	__shared__  unsigned char idx[32];
	out[1] = 0;
	if(threadIdx.z < 32){
		idx[threadIdx.z] = threadIdx.z;
	}
	__syncthreads();
	if(threadIdx.z < 32){
		if(kernel[threadIdx.z+32] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+32];
			idx[threadIdx.z] = threadIdx.z+32;
		}else
			idx[threadIdx.z] = threadIdx.z;
	}
	__syncthreads();
	if(threadIdx.z < 16){
		if(kernel[threadIdx.z+16] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+16];
			idx[threadIdx.z] = idx[threadIdx.z+16];
		}
	}
	__syncthreads();
	if(threadIdx.z < 8){
		if(kernel[threadIdx.z+8] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+8];
			idx[threadIdx.z] = idx[threadIdx.z+8];
		}
	}
	__syncthreads();
	if(threadIdx.z < 4){
		if(kernel[threadIdx.z+4] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+4];
			idx[threadIdx.z] = idx[threadIdx.z+4];
		}
	}
	__syncthreads();
	if(threadIdx.z < 2){
		if(kernel[threadIdx.z+2] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+2];
			idx[threadIdx.z] = idx[threadIdx.z+2];
		}
	}
	__syncthreads();
	if(threadIdx.z < 1){
		if(kernel[threadIdx.z+1] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+1];
			idx[threadIdx.z] = idx[threadIdx.z+1];
		}
		if(kernel[threadIdx.z] < *th && idx[threadIdx.z] < 64)
			out[0] = idx[threadIdx.z];
		else
			out[0] = 0;
	}
}

__device__ int abss(int a)
{
  return max(-a, a);
}

__global__ void brain1(const int rows, const int cols, unsigned char *edgeL, unsigned int* i0, unsigned int* i1,unsigned int* i2,unsigned int* i3,unsigned int* i4, unsigned int* i0x, unsigned int* i1x,unsigned int* i2x,unsigned int* i3x,unsigned int* i4x, unsigned int* i0y, unsigned int* i1y,unsigned int* i2y,unsigned int* i3y,unsigned int* i4y, unsigned int* i0xy, unsigned int* i1xy,unsigned int* i2xy,unsigned int* i3xy,unsigned int* i4xy, unsigned int *weights, unsigned char *disp){

	__shared__  unsigned int kernel[64];
	__shared__  unsigned char out[1];
	extern __shared__ unsigned int s16[]; 
	
	int t = blockIdx.y*cols+blockIdx.x;
	int th = 200000;
	
	/*const int weights[32] = {300,300,300,300,
						     300,300,300,300,
						     250,250,250,250,
							 200,200,200,200,
						     150,150,150,150,
							 120,120,120,120,
						     100,100,100,100,
							 100,100,100,100};*/
	float x32 = 1.0-((float)(abss(blockIdx.x%32-16))/16);
	float xx32 = 1.0-x32; 
	float y32 = 1.0-((float)(abss(blockIdx.y%32-16))/16);
	float yy32 = 1.0-y32; 
	float x16 = 1.0-((float)(abss(blockIdx.x%16-8))/8);
	float xx16 = 1.0-x16; 
	float y16 = 1.0-((float)(abss(blockIdx.y%16-8))/8);
	float yy16 = 1.0-y16; 

	
	__syncthreads();
	int distance = edgeL[t];
	//if(edgeL[t]){
	if(distance >= 32) return;
	
/*weights[distance]*/
			kernel[threadIdx.z] = weights[distance]*i0[(blockIdx.y/2*(320)+blockIdx.x/2)*64+threadIdx.z];
			//kernel[threadIdx.z] += y4*x4*4*i1[(blockIdx.y/4*(160)+blockIdx.x/4)*64+threadIdx.z];
			//kernel[threadIdx.z] += y8*x8*2*i2[(blockIdx.y/8*(80)+blockIdx.x/8)*64+threadIdx.z];
			kernel[threadIdx.z] += y16*x16*i3[(blockIdx.y/16*(40)+blockIdx.x/16)*64+threadIdx.z];
			kernel[threadIdx.z] += y32*x32*i4[(blockIdx.y/32*(20)+blockIdx.x/32)*64+threadIdx.z]/4;			
			if(blockIdx.x >= 16 && blockIdx.x <= cols-16) {
				kernel[threadIdx.z] += weights[distance]*i0x[(blockIdx.y/2*(320)+(blockIdx.x-1)/2)*64+threadIdx.z];
				//kernel[threadIdx.z] += y4*xx4*4*i1x[((blockIdx.y)/4*(160)+(blockIdx.x-2)/4)*64+threadIdx.z];
				//kernel[threadIdx.z] += y8*xx8*2*i2x[((blockIdx.y)/8*(80)+(blockIdx.x-4)/8)*64+threadIdx.z];
				kernel[threadIdx.z] += y16*xx16*i3[((blockIdx.y)/16*(40)+(blockIdx.x-8)/16)*64+threadIdx.z];
				kernel[threadIdx.z] += y32*xx32*i4y[((blockIdx.y)/32*(20)+(blockIdx.x-16)/32)*64+threadIdx.z]/4;
			}
			if(blockIdx.y >= 16 && blockIdx.y <= rows-16) {
				kernel[threadIdx.z] += weights[distance]*i0y[((blockIdx.y-1)/2*(320)+blockIdx.x/2)*64+threadIdx.z];
				//kernel[threadIdx.z] += yy4*x4*4*i1y[((blockIdx.y-2)/4*(160)+(blockIdx.x)/4)*64+threadIdx.z];
				//kernel[threadIdx.z] += yy8*x8*2*i2y[((blockIdx.y-4)/8*(80)+(blockIdx.x)/8)*64+threadIdx.z];
				kernel[threadIdx.z] += yy16*x16*i3[((blockIdx.y-8)/16*(40)+(blockIdx.x)/16)*64+threadIdx.z];
				kernel[threadIdx.z] += yy32*x32*i4y[((blockIdx.y-16)/32*(20)+(blockIdx.x)/32)*64+threadIdx.z]/4;
				
			}
			if(blockIdx.y >= 16 && blockIdx.y <= rows-16 && blockIdx.x >= 16 && blockIdx.x <= cols-16) {
				kernel[threadIdx.z] += weights[distance]*i0xy[((blockIdx.y-1)/2*(320)+(blockIdx.x-1)/2)*64+threadIdx.z];
				//kernel[threadIdx.z] += yy4*xx4*4*i1xy[((blockIdx.y-2)/4*(160)+(blockIdx.x-2)/4)*64+threadIdx.z];
				//kernel[threadIdx.z] += yy8*xx8*2*i2xy[((blockIdx.y-4)/8*(80)+(blockIdx.x-4)/8)*64+threadIdx.z];
				kernel[threadIdx.z] += yy16*xx16*i3xy[((blockIdx.y-8)/16*(40)+(blockIdx.x-8)/16)*64+threadIdx.z];
				kernel[threadIdx.z] += yy32*xx32*i4xy[((blockIdx.y-16)/32*(20)+(blockIdx.x-16)/32)*64+threadIdx.z]/4;
			}
		__syncthreads();
		findMin64(kernel, out, &th);
		__syncthreads();
		if(threadIdx.z == 0){
			//if(out[0] == 0){
			//	disp[t] = 0;
			//}else
				disp[t] = out[0]*4;
			//}
	}//else{
	//	if(threadIdx.z == 0)
	//		disp[t] = 0;
	//}
}

__global__ void brain11(const int rows, const int cols, unsigned char *edgeL, unsigned short* i8, unsigned short *weights, unsigned char *disp){

	__shared__  unsigned int kernel[64];
	__shared__  unsigned char out[1];
	extern __shared__ unsigned int s16[]; 
	
	int t = blockIdx.y*cols+blockIdx.x;
	int th = 200000;
	
	
	__syncthreads();
	int distance = edgeL[t];
	//if(edgeL[t]){
	if(distance >= 32) return;
	
/*weights[distance]*/

		kernel[threadIdx.z] = i8[(blockIdx.y/8*(80)+blockIdx.x/8)*64+threadIdx.z];	
		__syncthreads();
		findMin64(kernel, out, &th);
		__syncthreads();
		if(threadIdx.z == 0){
			//if(out[0] == 0){
			//	disp[t] = 0;
			//}else
				disp[t] = out[0]*4;
			//}
	}//else{
	//	if(threadIdx.z == 0)
	//		disp[t] = 0;
	//}
}
__device__ void findMin64US(unsigned short kernel[64], unsigned char out[1], int *th){
	__shared__  unsigned char idx[32];
	out[1] = 0;
	if(threadIdx.z < 32){
		idx[threadIdx.z] = threadIdx.z;
	}
	__syncthreads();
	if(threadIdx.z < 32){
		if(kernel[threadIdx.z+32] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+32];
			idx[threadIdx.z] = threadIdx.z+32;
		}else
			idx[threadIdx.z] = threadIdx.z;
	}
	__syncthreads();
	if(threadIdx.z < 16){
		if(kernel[threadIdx.z+16] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+16];
			idx[threadIdx.z] = idx[threadIdx.z+16];
		}
	}
	__syncthreads();
	if(threadIdx.z < 8){
		if(kernel[threadIdx.z+8] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+8];
			idx[threadIdx.z] = idx[threadIdx.z+8];
		}
	}
	__syncthreads();
	if(threadIdx.z < 4){
		if(kernel[threadIdx.z+4] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+4];
			idx[threadIdx.z] = idx[threadIdx.z+4];
		}
	}
	__syncthreads();
	if(threadIdx.z < 2){
		if(kernel[threadIdx.z+2] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+2];
			idx[threadIdx.z] = idx[threadIdx.z+2];
		}
	}
	__syncthreads();
	if(threadIdx.z < 1){
		if(kernel[threadIdx.z+1] < kernel[threadIdx.z]){
			kernel[threadIdx.z] = kernel[threadIdx.z+1];
			idx[threadIdx.z] = idx[threadIdx.z+1];
		}
		if(kernel[threadIdx.z] < *th && idx[threadIdx.z] < 64)
			out[0] = idx[threadIdx.z];
		else
			out[0] = 0;
	}
}

__global__ void brain2(const int rows, const int cols, unsigned char *edgeL, unsigned short* in8, unsigned short *weights, unsigned char *disp){

	//__shared__  unsigned short kernel[64];
	//__shared__  unsigned char out[1];
	__shared__  unsigned char pattern[18][18];
	__shared__  unsigned char search[64+18][18];
	extern __shared__ unsigned int extt[];
	
	unsigned int *block32 = (unsigned int*)&extt[0];
	//size of block32 is 3x3x64
	
	unsigned int *block16 = (unsigned int*)&block32[3*3*64];
	//size of block 16 is 2x2x64
	
	unsigned short *block8 = (unsigned short*)&block16[2*2*64];
	//size of block8 is 6x6x64
	
//------fill block8----------//
	if(blockIdx.x < 2 || blockIdx.x > (cols/8)-1 || blockIdx.y < 2 || blockIdx.y > (rows/8)-1) return; 
	int block8idxX = 2 + blockIdx.x%2;
	int block8idxY = 2 + blockIdx.y%2;
	int block8idx = ((blockIdx.y+(threadIdx.y-block8idxY))*(cols/8)+blockIdx.x+(threadIdx.x-block8idxX))*64+threadIdx.z;

	int thread8idx = 64*(threadIdx.y*6+threadIdx.x)+threadIdx.z;
	
	if(threadIdx.x < 6 && threadIdx.y < 6){
		block8[thread8idx] = in8[block8idx];
		block8[thread8idx+16] = in8[block8idx+16];
		block8[thread8idx+32] = in8[block8idx+32];
		block8[thread8idx+48] = in8[block8idx+48];
	}
	__syncthreads();
//------calculate ----------//	
	pattern[threadIdx.x+1][threadIdx.y+1] = 0;
	search[threadIdx.x+1][threadIdx.y+1] = pattern[threadIdx.x+1][threadIdx.y+1];
	pattern[threadIdx.x+1][threadIdx.y+1] = search[threadIdx.x+1][threadIdx.y+1];
	//int th = 200000;
	
	
	__syncthreads();
	/*
	if(distance >= 32) return;
	
			//kernel[threadIdx.z] = in8[idx8];
		__syncthreads();
		findMin64US(kernel, out, &th);
		__syncthreads();
		if(threadIdx.z == 0){
				disp[t] = out[0]*4;
	}
	*/
}
__device__ void match2extend_16x16x2(unsigned char pattern[18][18], unsigned char search[64+18][18], unsigned short *block2, const unsigned short shift){
	
	short iX = threadIdx.x+1;
	short iY = threadIdx.y+1;
	int idx = ((threadIdx.y * 16 + threadIdx.x)*64) + threadIdx.z + shift;
	
	block2[idx] =    absus((short)pattern[iX-1][iY-1] - search[iX-1+shift+threadIdx.z][iY-1]);
	block2[idx] += 2*absus((short)pattern[iX  ][iY-1] - search[iX  +shift+threadIdx.z][iY-1]);
	block2[idx] +=   absus((short)pattern[iX+1][iY-1] - search[iX+1+shift+threadIdx.z][iY-1]);
	block2[idx] += 2*absus((short)pattern[iX-1][iY  ] - search[iX-1+shift+threadIdx.z][iY  ]);
	block2[idx] += 4*absus((short)pattern[iX  ][iY  ] - search[iX  +shift+threadIdx.z][iY  ]);
	block2[idx] += 2*absus((short)pattern[iX+1][iY  ] - search[iX+1+shift+threadIdx.z][iY  ]);
	block2[idx] +=   absus((short)pattern[iX-1][iY+1] - search[iX-1+shift+threadIdx.z][iY+1]);
	block2[idx] += 2*absus((short)pattern[iX  ][iY+1] - search[iX  +shift+threadIdx.z][iY+1]);
	block2[idx] +=   absus((short)pattern[iX+1][iY+1] - search[iX+1+shift+threadIdx.z][iY+1]);
	
	block2[idx+2] =    absus((short)pattern[iX-1][iY-1] - search[iX-1+shift+threadIdx.z+2][iY-1]);
	block2[idx+2] += 2*absus((short)pattern[iX  ][iY-1] - search[iX  +shift+threadIdx.z+2][iY-1]);
	block2[idx+2] +=   absus((short)pattern[iX+1][iY-1] - search[iX+1+shift+threadIdx.z+2][iY-1]);
	block2[idx+2] += 2*absus((short)pattern[iX-1][iY  ] - search[iX-1+shift+threadIdx.z+2][iY  ]);
	block2[idx+2] += 4*absus((short)pattern[iX  ][iY  ] - search[iX  +shift+threadIdx.z+2][iY  ]);
	block2[idx+2] += 2*absus((short)pattern[iX+1][iY  ] - search[iX+1+shift+threadIdx.z+2][iY  ]);
	block2[idx+2] +=   absus((short)pattern[iX-1][iY+1] - search[iX-1+shift+threadIdx.z+2][iY+1]);
	block2[idx+2] += 2*absus((short)pattern[iX  ][iY+1] - search[iX  +shift+threadIdx.z+2][iY+1]);
	block2[idx+2] +=   absus((short)pattern[iX+1][iY+1] - search[iX+1+shift+threadIdx.z+2][iY+1]);
	
	block2[idx+4] =    absus((short)pattern[iX-1][iY-1] - search[iX-1+shift+threadIdx.z+4][iY-1]);
	block2[idx+4] += 2*absus((short)pattern[iX  ][iY-1] - search[iX  +shift+threadIdx.z+4][iY-1]);
	block2[idx+4] +=   absus((short)pattern[iX+1][iY-1] - search[iX+1+shift+threadIdx.z+4][iY-1]);
	block2[idx+4] += 2*absus((short)pattern[iX-1][iY  ] - search[iX-1+shift+threadIdx.z+4][iY  ]);
	block2[idx+4] += 4*absus((short)pattern[iX  ][iY  ] - search[iX  +shift+threadIdx.z+4][iY  ]);
	block2[idx+4] += 2*absus((short)pattern[iX+1][iY  ] - search[iX+1+shift+threadIdx.z+4][iY  ]);
	block2[idx+4] +=   absus((short)pattern[iX-1][iY+1] - search[iX-1+shift+threadIdx.z+4][iY+1]);
	block2[idx+4] += 2*absus((short)pattern[iX  ][iY+1] - search[iX  +shift+threadIdx.z+4][iY+1]);
	block2[idx+4] +=   absus((short)pattern[iX+1][iY+1] - search[iX+1+shift+threadIdx.z+4][iY+1]);
	
	
	block2[idx+6] =    absus((short)pattern[iX-1][iY-1] - search[iX-1+shift+threadIdx.z+6][iY-1]);
	block2[idx+6] += 2*absus((short)pattern[iX  ][iY-1] - search[iX  +shift+threadIdx.z+6][iY-1]);
	block2[idx+6] +=   absus((short)pattern[iX+1][iY-1] - search[iX+1+shift+threadIdx.z+6][iY-1]);
	block2[idx+6] += 2*absus((short)pattern[iX-1][iY  ] - search[iX-1+shift+threadIdx.z+6][iY  ]);
	block2[idx+6] += 4*absus((short)pattern[iX  ][iY  ] - search[iX  +shift+threadIdx.z+6][iY  ]);
	block2[idx+6] += 2*absus((short)pattern[iX+1][iY  ] - search[iX+1+shift+threadIdx.z+6][iY  ]);
	block2[idx+6] +=   absus((short)pattern[iX-1][iY+1] - search[iX-1+shift+threadIdx.z+6][iY+1]);
	block2[idx+6] += 2*absus((short)pattern[iX  ][iY+1] - search[iX  +shift+threadIdx.z+6][iY+1]);
	block2[idx+6] +=   absus((short)pattern[iX+1][iY+1] - search[iX+1+shift+threadIdx.z+6][iY+1]);
	
}

__device__ void findBestDisp(unsigned int *in, unsigned char *indexes){

	//short inIdx = 2*(4*mod(threadIdx.y,4)+mod(threadIdx.x,4))+threadIdx.z;
	/*short i = 4*(threadIdx.y/4)+(threadIdx.x/4);
	short inIdx = 2*(4*(threadIdx.y&(3))+(threadIdx.x&3))+threadIdx.z;
	short idxRes = i*64+inIdx;
	short idxIdx = i*32+inIdx;*/
	int inIdx = threadIdx.x + threadIdx.z*16; // after optimalisation
	int idxRes = threadIdx.y * 64 + inIdx;
	int idxIdx = threadIdx.y * 32 + inIdx;

		
	if(in[idxRes+32] < in[idxRes]){
			in[idxRes] = in[idxRes+32];
			indexes[idxIdx] = inIdx+32;
		}else
			indexes[idxIdx] = inIdx;
	__syncthreads();
	if(inIdx < 16){
		if(in[idxRes+16] < in[idxRes]){
			in[idxRes] = in[idxRes+16];
			indexes[idxIdx] = indexes[idxIdx+16];
		}
	}
	__syncthreads();
	if(inIdx < 8){
		if(in[idxRes+8] < in[idxRes]){
			in[idxRes] = in[idxRes+8];
			indexes[idxIdx] = indexes[idxIdx+8];
		}
	}
	__syncthreads();
	if(inIdx < 4){
		if(in[idxRes+4] < in[idxRes]){
			in[idxRes] = in[idxRes+4];
			indexes[idxIdx] = indexes[idxIdx+4];
		}
	}
	__syncthreads();
	if(inIdx < 2){
		if(in[idxRes+2] < in[idxRes]){
			in[idxRes] = in[idxRes+2];
			indexes[idxIdx] = indexes[idxIdx+2];
		}
	}
	__syncthreads();
	if(inIdx == 0){
		if(in[idxRes+1] < in[idxRes]){
			in[idxRes] = in[idxRes+1];
			indexes[idxIdx] = indexes[idxIdx+1];
		}
	}
}
__global__ void brain3(const int rows, const int cols, unsigned char *left, unsigned char *right, unsigned char *edgeL, unsigned short* in8, unsigned short *weights, unsigned char *disp){
	//__shared__  unsigned short kernel[64];
	//__shared__  unsigned char out[1];
	__shared__  unsigned char pattern[18][18];
	__shared__  unsigned char search[64+18][18];
	__shared__ unsigned short w[32];
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int t = y*cols+x;
	
	extern __shared__ unsigned int extt[];
	
	unsigned int *block16 = (unsigned int*)&extt[0];
	//size of block 16 is 2x2x64
	
	unsigned short *block8 = (unsigned short*)&block16[3*3*64];
	//size of block8 is 6x6x64
	
	unsigned short *block2 = (unsigned short*)&block8[6*6*64];
	//size of block2 is 16x16x64
	
	unsigned int *res = (unsigned int*)&block2[16*16*64];
	
	unsigned char *indexes = (unsigned char*)&res[4*4*64];
	
	if(threadIdx.y == 0){
		w[threadIdx.x+16*threadIdx.z] = weights[threadIdx.x+16*threadIdx.z];
	}

	if(blockIdx.x < 1 || blockIdx.x >= (cols/16)-1 || blockIdx.y < 1 || blockIdx.y >= (rows/16)-1) return; 
//------fill pattern---------//
	int iX = threadIdx.x+1;
	int iY = threadIdx.y+1;
	if(threadIdx.z == 0){
		pattern[iX][iY] = left[t];
		if(iX < 2){
			if(x < 1){
				pattern[iX-1][iY] = 0;
	   			if(iY < 2) pattern[iX-1][iY-1] = 0;
	   		}else{
	   			pattern[iX-1][iY] = left[y*cols+x-1];
	   			if(y < 1) pattern[iX-1][iY-1] = 0;
				else pattern[iX-1][iY-1] = left[(y-1)*cols+(x-1)];  		
			}
	   	}
	   	if(iX >= blockDim.x){
	   		if(x >= cols-1){
	   			pattern[iX+1][iY] = 0;
				if(iY >= blockDim.y) pattern[iX+1][iY+1] = 0;
			}else{
				pattern[iX+1][iY] = left[y*cols+(x+1)];
	   			if(y >= rows-1) pattern[iX+1][iY+1] = 0;
				else pattern[iX+1][iY+1] = left[(y+1)*cols+x+1];
	   		}
	   	}
		if(iY < 2){
			if(y < 1){ 
				pattern[iX][iY-1] = 0;
				if(iX >= blockDim.x) pattern[iX+1][iY-1] = 0;
			}else{
				pattern[iX][iY-1] = left[(y-1)*cols+x];
				if(x >= cols-1) pattern[iX+1][iY-1] = 0;
				else pattern[iX+1][iY-1] = left[(y-1)*cols+x+1];
			}
		}
	   	if(iY >= blockDim.y){
	   		if(y >= rows-1){
	   			pattern[iX][iY+1] = 0;
	   			if(iX < 2) pattern[iX-1][iY+1] = 0;
	   		}else{
	   			pattern[iX][iY+1] = left[(y+1)*cols+x];
	   			if(x < 1) pattern[iX-1][iY+1] = 0;
	   			else pattern[iX-1][iY+1] = left[(y+1)*cols+(x-1)];
	   		}
	   	}
	}
//------fill search----------//
	if(threadIdx.z == 0){
		search[iX][iY] = right[t];
		search[iX+16][iY] = right[t+16];
		if(threadIdx.x == 0){
			search[0][iY] = right[t-1];
		}
		if(threadIdx.y == 0){
			search[iX][0] = right[t-cols];
			search[iX+16][0] = right[t-cols+16];
			if(threadIdx.x == 0){
				search[0][0] = right[t-cols-1];
			}
			if(threadIdx.x == 15){
				search[iX+64+1][0] = right[t-cols+64+1];
			}
		}
		if(threadIdx.y == 15){
			search[iX][17] = right[t+cols];
			search[iX+16][17] = right[t+cols+16];
			if(threadIdx.x == 0){
				search[0][17] = right[t+cols-1];
			}
			if(threadIdx.x == 15){
				search[iX+64+1][17] = right[t+cols+64+1];
			}
		}
	}
	if(threadIdx.z == 1){
		search[iX+32][iY] = right[t+32];
		search[iX+48][iY] = right[t+48];
		search[iX+64][iY] = right[t+64];
		if(threadIdx.x == 15){
			search[64+17][iY] = right[t+64+1];
		}
		if(threadIdx.y == 0){
			search[iX+32][0] = right[t-cols+32];
			search[iX+48][0] = right[t-cols+48];
			search[iX+64][0] = right[t-cols+64];
		}
		if(threadIdx.y == 15){
			search[iX+32][17] = right[t+cols+32];
			search[iX+48][17] = right[t+cols+48];
			search[iX+64][17] = right[t+cols+64];
		}
	}

//------fill block8----------//
	int block8idx = 0;
	int thread8idx = 0;
	
	//spliting coping z axis between x and y threads
	if(threadIdx.x < 6 && threadIdx.y < 6){
		block8idx = ((2*blockIdx.y+(((int)threadIdx.y)-2))*(cols/8)+2*blockIdx.x+(((int)threadIdx.x)-2))*64+threadIdx.z;
		thread8idx = 64*(threadIdx.y*6+threadIdx.x)+threadIdx.z;
	}
	if(threadIdx.x >= 6 && threadIdx.x < 12 && threadIdx.y < 6){
		block8idx = ((2*blockIdx.y+(((int)threadIdx.y)-2))*(cols/8)+2*blockIdx.x+((((int)threadIdx.x)-6)-2))*64+threadIdx.z+16;
		thread8idx = 64*(threadIdx.y*6+((int)threadIdx.x)-6)+threadIdx.z+16;
	}
	if(threadIdx.x < 6 && threadIdx.y >= 6 && threadIdx.y < 12){
		block8idx = ((2*blockIdx.y+(((int)threadIdx.y)-8))*(cols/8)+2*blockIdx.x+(((int)threadIdx.x)-2))*64+threadIdx.z+32;
		thread8idx = 64*((((int)threadIdx.y)-6)*6+threadIdx.x)+threadIdx.z+32;
	}
	if(threadIdx.x >= 6 && threadIdx.x < 12 && threadIdx.y >= 6 && threadIdx.y < 12){
		block8idx = ((2*blockIdx.y+(((int)threadIdx.y)-8))*(cols/8)+2*blockIdx.x+((((int)threadIdx.x)-6)-2))*64+threadIdx.z+48;
		thread8idx = 64*((((int)threadIdx.y)-6)*6+((int)threadIdx.x)-6)+threadIdx.z+48;
	}
	
	if(threadIdx.x < 12 && threadIdx.y < 12){
		block8[thread8idx   ] = in8[block8idx   ];
		block8[thread8idx+2 ] = in8[block8idx+2 ];
		block8[thread8idx+4 ] = in8[block8idx+4 ];
		block8[thread8idx+6 ] = in8[block8idx+6 ];
		block8[thread8idx+8 ] = in8[block8idx+8 ];
		block8[thread8idx+10] = in8[block8idx+10];
		block8[thread8idx+12] = in8[block8idx+12];
		block8[thread8idx+14] = in8[block8idx+14];
	}
	__syncthreads();

//------calculate 2x2 extended blocks-----------//

	match2extend_16x16x2(pattern, search, block2, 0);
	match2extend_16x16x2(pattern, search, block2, 8);
	match2extend_16x16x2(pattern, search, block2, 16);
	match2extend_16x16x2(pattern, search, block2, 24);
	match2extend_16x16x2(pattern, search, block2, 32);
	match2extend_16x16x2(pattern, search, block2, 40);
	match2extend_16x16x2(pattern, search, block2, 48);
	match2extend_16x16x2(pattern, search, block2, 56);
	

//----copy edgeDistanceTransform to pattern-----------//
	if(threadIdx.z == 0)
		pattern[threadIdx.x][threadIdx.y] = edgeL[t];

//------calculate 16x16 blocks----------//
	int b16idx = ((threadIdx.y/4)*3+(threadIdx.x/4))*64;
	int b8idx  = ((threadIdx.y/4)*6+(threadIdx.x/4))*64*2;
	int zidx = 2*(4*(threadIdx.y%4) + (threadIdx.x%4))+threadIdx.z;
	
	if(threadIdx.x < 12 && threadIdx.y < 12){
		block16[b16idx+zidx]   = block8[b8idx+zidx];
		block16[b16idx+zidx+32] = block8[b8idx+zidx+32];
		
		block16[b16idx+zidx]   += block8[b8idx+zidx+64];
		block16[b16idx+zidx+32] += block8[b8idx+zidx+32+64];
		
		block16[b16idx+zidx]   += block8[b8idx+zidx+64*6];
		block16[b16idx+zidx+32] += block8[b8idx+zidx+32+64*6];
		
		block16[b16idx+zidx]   += block8[b8idx+zidx+64*7];
		block16[b16idx+zidx+32] += block8[b8idx+zidx+32+64*7];
		
	}
	__syncthreads();
	
//----calculate(3x3) 16x16 blocks into (3x) 32x32 blocks with results in corners

	zidx = (2*(4*(threadIdx.y%8)+(threadIdx.x%4)))+threadIdx.z;
	
	if(threadIdx.x < 4 && threadIdx.y < 8){
		block16[0+zidx] += block16[64+zidx];
		block16[0+zidx] += block16[64*3+zidx];
		block16[0+zidx] += block16[64*4+zidx];
	}
	if(threadIdx.x < 4 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[64*2+zidx] += block16[64+zidx];
		block16[64*2+zidx] += block16[64*4+zidx];
		block16[64*2+zidx] += block16[64*5+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y < 8){
		block16[64*6+zidx] += block16[64*3+zidx];
		block16[64*6+zidx] += block16[64*4+zidx];
		block16[64*6+zidx] += block16[64*7+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[64*8+zidx] += block16[64*4+zidx];
		block16[64*8+zidx] += block16[64*5+zidx];
		block16[64*8+zidx] += block16[64*7+zidx];
	}
	__syncthreads();	
	
	//int shift = 2*(4*(threadIdx.y%4)+(threadIdx.x%4))+threadIdx.z;
	//int idxRes = (4*(threadIdx.y/4)+(threadIdx.x/4))*64+shift;
	int shift = 16*threadIdx.z + threadIdx.x; 
	int idxRes = threadIdx.y*64 + shift;

	int idxB32_1 = 0*64+shift;
	int idxB32_2 = 2*64+shift;
	int idxB32_3 = 6*64+shift;
	int idxB32_4 = 8*64+shift;
	int idxB16_1 = 3*64+shift;
	int idxB16_2 = 1*64+shift;
	int idxB16_3 = 7*64+shift;
	int idxB16_4 = 5*64+shift;
///--------------FIND RESULTS FOR FIRST Q --------------///                             1111111111111111111111
//----calculate(3x3) 16x16 blocks into (3x3) 16x16 blocks with results in cross
	if(threadIdx.x < 4 && threadIdx.y < 8){
		block16[1*64+zidx] =  block8[8*64+zidx];
		block16[1*64+zidx] += block8[9*64+zidx];
		block16[1*64+zidx] += block8[14*64+zidx];
		block16[1*64+zidx] += block8[15*64+zidx];
	}
	if(threadIdx.x < 4 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[3*64+zidx] =  block8[7*64+zidx];
		block16[3*64+zidx] += block8[8*64+zidx];
		block16[3*64+zidx] += block8[13*64+zidx];
		block16[3*64+zidx] += block8[14*64+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y < 8){
		block16[64*5+zidx] =  block8[64*14+zidx];
		block16[64*5+zidx] += block8[64*15+zidx];
		block16[64*5+zidx] += block8[64*20+zidx];
		block16[64*5+zidx] += block8[64*21+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[64*7+zidx] =  block8[64*13+zidx];
		block16[64*7+zidx] += block8[64*14+zidx];
		block16[64*7+zidx] += block8[64*19+zidx];
		block16[64*7+zidx] += block8[64*20+zidx];
	}

	if(blockIdx.x < 1 || blockIdx.x >= (cols/16)-4 || blockIdx.y < 1 || blockIdx.y >= (rows/16)-1) return;

//-----sum blocks results 16+32+2 ----// 1;1
	int thYmod = threadIdx.y%4;
	int thYdiv = threadIdx.y/4;
	float xx32 = ((float)(0+(thYmod)))/16.0;// for next iter 0 change to 4/8/12
	float yy32 = ((float)(0+(thYdiv)))/16.0;// for next iter 0 change to 4/8/12
	float xx16 = ((float)(0+(thYmod)))/8.0;// for next iter 0 change to 4/8/12
	float yy16 = ((float)(0+(thYdiv)))/8.0;// for next iter 0 change to 4/8/12
	float x32  = 1.0-xx32;
	float y32  = 1.0-yy32;
	float x16  = 1.0-xx16;
	float y16  = 1.0-yy16;
	

	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	unsigned short idx2 = ((0+(thYdiv))*16+(0+(thYmod)))*64+shift;
	unsigned short weight = w[pattern[0+(thYmod)][0+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
		
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x < 4 && threadIdx.y < 4 && threadIdx.z == 0){
		disp[t] = 4*indexes[(threadIdx.y*4+threadIdx.x)*32];
	}	



//-----sum blocks results 16+32+2 ----// 2;1
	xx32 = ((float)(4+(thYmod)))/16.0;// for next iter 0 change to 4/8/12
	yy32 = ((float)(0+(thYdiv)))/16.0;// for next iter 0 change to 4/8/12
	xx16 = ((float)(4+(thYmod)))/8.0;// for next iter 0 change to 4/8/12
	yy16 = ((float)(0+(thYdiv)))/8.0;// for next iter 0 change to 4/8/12
	x32  = 1.0-xx32;
	y32  = 1.0-yy32;
	x16  = 1.0-xx16;
	y16  = 1.0-yy16;
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((0+(thYdiv))*16+(4+(thYmod)))*64+shift;
	weight = w[pattern[4+(thYmod)][0+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y < 4 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.y%4))*32];
	}
	
	__syncthreads();	
	
//-----sum blocks results 16+32+2 ----// 1;2
	xx32 = ((float)(0+(thYmod)))/16.0;// for next iter 0 change to 4/8/12
	yy32 = ((float)(4+(thYdiv)))/16.0;// for next iter 0 change to 4/8/12
	xx16 = ((float)(0+(thYmod)))/8.0;// for next iter 0 change to 4/8/12
	yy16 = ((float)(4+(thYdiv)))/8.0;// for next iter 0 change to 4/8/12
	x32  = 1.0-xx32;
	y32  = 1.0-yy32;
	x16  = 1.0-xx16;
	y16  = 1.0-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((4+(thYdiv))*16+(0+(thYmod)))*64+shift;
	weight = w[pattern[0+(thYmod)][4+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();	
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x < 4 && threadIdx.y >= 4 && threadIdx.y < 8 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}
//-----sum blocks results 16+32+2 ----// 2;2
	xx32 = ((float)(4+(thYmod)))/16.0;// for next iter 0 change to 4/8/12
	yy32 = ((float)(4+(thYdiv)))/16.0;// for next iter 0 change to 4/8/12
	xx16 = ((float)(4+(thYmod)))/8.0;// for next iter 0 change to 4/8/12
	yy16 = ((float)(4+(thYdiv)))/8.0;// for next iter 0 change to 4/8/12
	x32  = 1.0-xx32;
	y32  = 1.0-yy32;
	x16  = 1.0-xx16;
	y16  = 1.0-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((4+(thYdiv))*16+(4+(thYmod)))*64+shift;
	weight = w[pattern[4+(thYmod)][4+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y >= 4 && threadIdx.y < 8 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}
	
	
	

///--------------FIND RESULTS FOR SECOND Q --------------///                            22222222222222222222222222
//----calculate(3x3) 16x16 blocks into (3x3) 16x16 blocks with results in cross
	if(threadIdx.x < 4 && threadIdx.y < 8){
		block16[1*64+zidx] =  block8[9*64+zidx];
		block16[1*64+zidx] += block8[10*64+zidx];
		block16[1*64+zidx] += block8[15*64+zidx];
		block16[1*64+zidx] += block8[16*64+zidx];
	}
	if(threadIdx.x < 4 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[3*64+zidx] =  block8[8*64+zidx];
		block16[3*64+zidx] += block8[9*64+zidx];
		block16[3*64+zidx] += block8[14*64+zidx];
		block16[3*64+zidx] += block8[15*64+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y < 8){
		block16[64*5+zidx] =  block8[64*15+zidx];
		block16[64*5+zidx] += block8[64*16+zidx];
		block16[64*5+zidx] += block8[64*21+zidx];
		block16[64*5+zidx] += block8[64*22+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[64*7+zidx] =  block8[64*14+zidx];
		block16[64*7+zidx] += block8[64*15+zidx];
		block16[64*7+zidx] += block8[64*20+zidx];
		block16[64*7+zidx] += block8[64*21+zidx];
	}

//-----sum blocks results 16+32+2 ----// 1;1

	xx32 = ((float)(8+(thYmod)))/16.0;// for next iter 0 change to 4/8/12
	yy32 = ((float)(0+(thYdiv)))/16.0;// for next iter 0 change to 4/8/12
	xx16 = ((float)(0+(thYmod)))/8.0;// for next iter 0 change to 4/8/12
	yy16 = ((float)(0+(thYdiv)))/8.0;// for next iter 0 change to 4/8/12
	x32  = 1.0-xx32;
	y32  = 1.0-yy32;
	x16  = 1.0-xx16;
	y16  = 1.0-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((0+(thYdiv))*16+(8+(thYmod)))*64+shift;
	weight = w[pattern[8+(thYmod)][0+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 8 && threadIdx.x < 12 && threadIdx.y < 4 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}	
//-----sum blocks results 16+32+2 ----// 2;1
	xx32 = ((float)(12+(thYmod)))/16.0;// for next iter 0 change to 4/8/12
	yy32 = ((float)(0+(thYdiv)))/16.0;// for next iter 0 change to 4/8/12
	xx16 = ((float)(4+(thYmod)))/8.0;// for next iter 0 change to 4/8/12
	yy16 = ((float)(0+(thYdiv)))/8.0;// for next iter 0 change to 4/8/12
	x32  = 1.0-xx32;
	y32  = 1.0-yy32;
	x16  = 1.0-xx16;
	y16  = 1.0-yy16;
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((0+(thYdiv))*16+(12+(thYmod)))*64+shift;
	weight = w[pattern[12+(thYmod)][0+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 12 && threadIdx.x < 16 && threadIdx.y < 4 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}
	
//-----sum blocks results 16+32+2 ----// 1;2
	xx32 = ((float)(8+(thYmod)))/16;// for next iter 0 change to 4/8/12
	yy32 = ((float)(4+(thYdiv)))/16;// for next iter 0 change to 4/8/12
	xx16 = ((float)(0+(thYmod)))/8;// for next iter 0 change to 4/8/12
	yy16 = ((float)(4+(thYdiv)))/8;// for next iter 0 change to 4/8/12
	x32  = 1-xx32;
	y32  = 1-yy32;
	x16  = 1-xx16;
	y16  = 1-yy16;
		
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((4+(thYdiv))*16+(8+(thYmod)))*64+shift;
	weight = w[pattern[8+(thYmod)][4+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();	
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 8 && threadIdx.x < 12 && threadIdx.y >= 4 && threadIdx.y < 8 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}
//-----sum blocks results 16+32+2 ----// 2;2
	xx32 = ((float)(12+(thYmod)))/16.0;// for next iter 0 change to 4/8/12
	yy32 = ((float)(4+(thYdiv)))/16.0;// for next iter 0 change to 4/8/12
	xx16 = ((float)(4+(thYmod)))/8.0;// for next iter 0 change to 4/8/12
	yy16 = ((float)(4+(thYdiv)))/8.0;// for next iter 0 change to 4/8/12
	x32  = 1.0-xx32;
	y32  = 1.0-yy32;
	x16  = 1.0-xx16;
	y16  = 1.0-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((4+(thYdiv))*16+(12+(thYmod)))*64+shift;
	weight = w[pattern[12+(thYmod)][4+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 12 && threadIdx.x < 16 && threadIdx.y >= 4 && threadIdx.y < 8 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}

	///--------------FIND RESULTS FOR SECOND Q --------------///                            3333333333333333333333333
//----calculate(3x3) 16x16 blocks into (3x3) 16x16 blocks with results in cross
	if(threadIdx.x < 4 && threadIdx.y < 8){
		block16[1*64+zidx] =  block8[14*64+zidx];
		block16[1*64+zidx] += block8[15*64+zidx];
		block16[1*64+zidx] += block8[20*64+zidx];
		block16[1*64+zidx] += block8[21*64+zidx];
	}
	if(threadIdx.x < 4 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[3*64+zidx] =  block8[13*64+zidx];
		block16[3*64+zidx] += block8[14*64+zidx];
		block16[3*64+zidx] += block8[19*64+zidx];
		block16[3*64+zidx] += block8[20*64+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y < 8){
		block16[64*5+zidx] =  block8[64*20+zidx];
		block16[64*5+zidx] += block8[64*21+zidx];
		block16[64*5+zidx] += block8[64*26+zidx];
		block16[64*5+zidx] += block8[64*27+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[64*7+zidx] =  block8[64*19+zidx];
		block16[64*7+zidx] += block8[64*20+zidx];
		block16[64*7+zidx] += block8[64*25+zidx];
		block16[64*7+zidx] += block8[64*26+zidx];
	}

//-----sum blocks results 16+32+2 ----// 1;1

	xx32 = ((float)(0+(thYmod)))/16;// for next iter 0 change to 4/8/12
	yy32 = ((float)(8+(thYdiv)))/16;// for next iter 0 change to 4/8/12
	xx16 = ((float)(0+(thYmod)))/8;// for next iter 0 change to 4/8/12
	yy16 = ((float)(0+(thYdiv)))/8;// for next iter 0 change to 4/8/12
	x32  = 1-xx32;
	y32  = 1-yy32;
	x16  = 1-xx16;
	y16  = 1-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((8+(thYdiv))*16+(0+(thYmod)))*64+shift;
	weight = w[pattern[0+(thYmod)][8+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.y >= 8 && threadIdx.y < 12 && threadIdx.x < 4 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}	

//-----sum blocks results 16+32+2 ----// 2;1
	xx32 = ((float)(4+(thYmod)))/16;// for next iter 0 change to 4/8/12
	yy32 = ((float)(8+(thYdiv)))/16;// for next iter 0 change to 4/8/12
	xx16 = ((float)(4+(thYmod)))/8;// for next iter 0 change to 4/8/12
	yy16 = ((float)(0+(thYdiv)))/8;// for next iter 0 change to 4/8/12
	x32  = 1-xx32;
	y32  = 1-yy32;
	x16  = 1-xx16;
	y16  = 1-yy16;
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((8+(thYdiv))*16+(4+(thYmod)))*64+shift;
	weight = w[pattern[4+(thYmod)][8+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y >= 8 && threadIdx.y < 12 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}
	
//-----sum blocks results 16+32+2 ----// 1;2
	xx32 = ((float)(0+(thYmod)))/16;// for next iter 0 change to 4/8/12
	yy32 = ((float)(12+(thYdiv)))/16;// for next iter 0 change to 4/8/12
	xx16 = ((float)(0+(thYmod)))/8;// for next iter 0 change to 4/8/12
	yy16 = ((float)(4+(thYdiv)))/8;// for next iter 0 change to 4/8/12
	x32  = 1-xx32;
	y32  = 1-yy32;
	x16  = 1-xx16;
	y16  = 1-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((12+(thYdiv))*16+(0+(thYmod)))*64+shift;
	weight = w[pattern[0+(thYmod)][12+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();	
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.y >= 12 && threadIdx.y < 16 && threadIdx.x < 4 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}
//-----sum blocks results 16+32+2 ----// 2;2
	xx32 = ((float)(4+(thYmod)))/16;// for next iter 0 change to 4/8/12
	yy32 = ((float)(12+(thYdiv)))/16;// for next iter 0 change to 4/8/12
	xx16 = ((float)(4+(thYmod)))/8;// for next iter 0 change to 4/8/12
	yy16 = ((float)(4+(thYdiv)))/8;// for next iter 0 change to 4/8/12
	x32  = 1-xx32;
	y32  = 1-yy32;
	x16  = 1-xx16;
	y16  = 1-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((12+(thYdiv))*16+(4+(thYmod)))*64+shift;
	weight = w[pattern[4+(thYmod)][12+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.y >= 12 && threadIdx.y < 16 && threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}



	///--------------FIND RESULTS FOR SECOND Q --------------///                            44444444444444444444
//----calculate(3x3) 16x16 blocks into (3x3) 16x16 blocks with results in cross
	if(threadIdx.x < 4 && threadIdx.y < 8){
		block16[1*64+zidx] =  block8[15*64+zidx];
		block16[1*64+zidx] += block8[16*64+zidx];
		block16[1*64+zidx] += block8[21*64+zidx];
		block16[1*64+zidx] += block8[22*64+zidx];
	}
	if(threadIdx.x < 4 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[3*64+zidx] =  block8[14*64+zidx];
		block16[3*64+zidx] += block8[15*64+zidx];
		block16[3*64+zidx] += block8[20*64+zidx];
		block16[3*64+zidx] += block8[21*64+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y < 8){
		block16[64*5+zidx] =  block8[64*21+zidx];
		block16[64*5+zidx] += block8[64*22+zidx];
		block16[64*5+zidx] += block8[64*27+zidx];
		block16[64*5+zidx] += block8[64*28+zidx];
	}
	if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y >= 8 && threadIdx.y < 16 ){
		block16[64*7+zidx] =  block8[64*20+zidx];
		block16[64*7+zidx] += block8[64*21+zidx];
		block16[64*7+zidx] += block8[64*26+zidx];
		block16[64*7+zidx] += block8[64*27+zidx];
	}

//-----sum blocks results 16+32+2 ----// 1;1

	xx32 = ((float)(8+(thYmod)))/16;// for next iter 0 change to 4/8/12
	yy32 = ((float)(8+(thYdiv)))/16;// for next iter 0 change to 4/8/12
	xx16 = ((float)(0+(thYmod)))/8;// for next iter 0 change to 4/8/12
	yy16 = ((float)(0+(thYdiv)))/8;// for next iter 0 change to 4/8/12
	x32  = 1-xx32;
	y32  = 1-yy32;
	x16  = 1-xx16;
	y16  = 1-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((8+(thYdiv))*16+(8+(thYmod)))*64+shift;
	weight = w[pattern[8+(thYmod)][8+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 8 && threadIdx.x < 12 && threadIdx.y >= 8 && threadIdx.y < 12 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}	

//-----sum blocks results 16+32+2 ----// 2;1
	xx32 = ((float)(12+(thYmod)))/16;// for next iter 0 change to 4/8/12
	yy32 = ((float)(8+(thYdiv)))/16;// for next iter 0 change to 4/8/12
	xx16 = ((float)(4+(thYmod)))/8;// for next iter 0 change to 4/8/12
	yy16 = ((float)(0+(thYdiv)))/8;// for next iter 0 change to 4/8/12
	x32  = 1-xx32;
	y32  = 1-yy32;
	x16  = 1-xx16;
	y16  = 1-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((8+(thYdiv))*16+(12+(thYmod)))*64+shift;
	weight = w[pattern[12+(thYmod)][8+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 12 && threadIdx.x < 16 && threadIdx.y >= 8 && threadIdx.y < 12 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}
	
//-----sum blocks results 16+32+2 ----// 1;2
	xx32 = ((float)(8+(thYmod)))/16;// for next iter 0 change to 4/8/12
	yy32 = ((float)(12+(thYdiv)))/16;// for next iter 0 change to 4/8/12
	xx16 = ((float)(0+(thYmod)))/8;// for next iter 0 change to 4/8/12
	yy16 = ((float)(4+(thYdiv)))/8;// for next iter 0 change to 4/8/12
	x32  = 1-xx32;
	y32  = 1-yy32;
	x16  = 1-xx16;
	y16  = 1-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((12+(thYdiv))*16+(8+(thYmod)))*64+shift;
	weight = w[pattern[8+(thYmod)][12+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	
	__syncthreads();	
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 8 && threadIdx.x < 12 && threadIdx.y >= 12 && threadIdx.y < 16 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}
//-----sum blocks results 16+32+2 ----// 2;2
	xx32 = ((float)(12+(thYmod)))/16;// for next iter 0 change to 4/8/12
	yy32 = ((float)(12+(thYdiv)))/16;// for next iter 0 change to 4/8/12
	xx16 = ((float)(4+(thYmod)))/8;// for next iter 0 change to 4/8/12
	yy16 = ((float)(4+(thYdiv)))/8;// for next iter 0 change to 4/8/12
	x32  = 1-xx32;
	y32  = 1-yy32;
	x16  = 1-xx16;
	y16  = 1-yy16;
	
	res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
	res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
	res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
	res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
	
	idx2 = ((12+(thYdiv))*16+(12+(thYmod)))*64+shift;
	weight = w[pattern[12+(thYmod)][12+(thYdiv)]];
	res[idxRes] += weight*block2[idx2];
	res[idxRes+32] += weight*block2[idx2+32];
	__syncthreads();
//------find max of sums -------------//
	findBestDisp(res, indexes);
	__syncthreads();	
//------save best results into the file-----//
	if(threadIdx.x >= 12 && threadIdx.x < 16 && threadIdx.y >= 12 && threadIdx.y < 16 && threadIdx.z == 0){
		disp[t] = 4*indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32];
	}
}







__global__ void median(const int rows, const int cols, unsigned char *src, unsigned char *edge, unsigned char *med){

	
	int x = blockIdx.x * blockDim.y + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
    
	int median[9];
	if(x < 1 || x >= cols-1 || y < 1 || y >= rows-1) return;
	//if(edge[y*cols+x]){
	
		median[0] = src[(y-1)*cols+x-1];
		median[1] = src[(y-1)*cols+x];
		median[2] = src[(y-1)*cols+x+1];
		median[3] = src[(y)*cols+x-1];
		median[4] = src[(y)*cols+x+1];
		median[5] = src[(y+1)*cols+x-1];
		median[6] = src[(y+1)*cols+x];
		median[7] = src[(y+1)*cols+x+1];
		median[8] = src[(y)*cols+x];
	
		int tmp;
	   for( int i = 0; i < 9; i++ )
		{
		    for( int j = 0; j < 8; j++ )
		    {
		        if( median[ j ] > median[ j + 1 ] ){
		        	tmp = median[j];
		        	median[j] = median[j+1];
		        	median[j+1] = tmp;
		        }
		    }
		}
		med[y*cols+x]= median[4];//in[y*cols+x];//(unsigned char)median[8];
	//}else
		//med[y*cols+x]=0;
    
}

__global__ void findDistance(const int rows, const int cols, unsigned char *edge, unsigned char *out){
	
	int x = blockIdx.x * blockDim.y + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	
	__syncthreads();
	
	int r = 0;
	int ix;
	int iy1; 
		while(r < 21){
			for(ix = -r; ix <=r; ix++){
				iy1 = r-abs(ix);
				if(x >= ix && x < cols-ix && y >= iy1 && y < rows-iy1){
					if(edge[(y+iy1)*cols+x+ix]){
						int value = (unsigned char)sqrtf(ix*ix+iy1*iy1);
						if(value < 21) out[y*cols+x] = value;
						else out[y*cols+x] = 20;
						return;
					}
					if(edge[(y-iy1)*cols+x+ix]){
						int value = (unsigned char)sqrtf(ix*ix+iy1*iy1);
						if(value < 21) out[y*cols+x] = value;
						else out[y*cols+x] = 20;
						return;
					}
				}

				
			}
		r++;
	}
	
	out[y*cols+x] = 20;
	return;

}

__global__ void findDistanceFast(const int rows, const int cols, unsigned char *edge, unsigned char *out){
	
	extern __shared__ unsigned char s[];
	
	//short minDist = 441;
	
	int x = blockIdx.x * blockDim.y + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	
	int iX = threadIdx.x + 20;
	int iY = threadIdx.y + 20;
	
	short idx = iX+64*(iY);
	
	s[idx] = edge[y*cols+x];
	
    if(iX < 40){
    	if(x < 20){
    		s[iX-20+64*(iY)] = 0;
    		if(iY < 40) s[iX-20+64*(iY-20)] = 0;
    	}else{
    		s[iX-20+64*(iY)] = edge[y*cols+x-20];
    		if(y < 20) s[iX-20+64*(iY-20)] = 0;
    		else s[iX-20+64*(iY-20)] = edge[(y-20)*cols+(x-20)];  		
    	}
    }
   	if(iX >= blockDim.x){
   		if(x >= cols-20){
   			s[iX+20+64*(iY)] = 0;
   			if(iY >= blockDim.y) s[iX+20+64*(iY+20)] = 0;
   		}else{
   			s[iX+20+64*(iY)] = edge[y*cols+(x+20)];
   			if(y >= rows-20) s[iX+20+64*(iY+20)] = 0;
   			else s[iX+20+64*(iY+20)] = edge[(y+20)*cols+x+20];
   		}
   	}
   	if(iY < 40){
   		if(y < 20){ 
   			s[iX+64*(iY-20)] = 0;
   			if(iX >= blockDim.x) s[iX+20+64*(iY-20)] = 0;
   		}else{
   			s[iX+64*(iY-20)] = edge[(y-20)*cols+x];
   			if(x >= cols-20) s[iX+20+64*(iY-20)] = 0;
   			else s[iX+20+64*(iY-20)] = edge[(y-20)*cols+x+20];
   		}
   	}
   	if(iY >= blockDim.y){
   		if(y >= rows-20){
   			s[iX+64*(iY+20)] = 0;
   			if(iX < 40) s[iX-20+64*(iY+20)] = 0;
   		}else{
   			s[iX+64*(iY+20)] = edge[(y+20)*cols+x];
  			if(x < 20) s[iX-20+64*(iY+20)] = 0;
   			else s[iX-20+64*(iY+20)] = edge[(y+20)*cols+(x-20)];
   		}
   	}   	
   	__syncthreads();
   	short idx1 = (iX-12)+64*(iY-12);
   	short idx2 = (iX+12)+64*(iY-12);
   	short idx3 = (iX-12)+64*(iY+12);
   	short idx4 = (iX+12)+64*(iY+12);
   	unsigned char i1 = 0;
   	unsigned char i2 = 0;
   	unsigned char i3 = 0;
   	unsigned char i4 = 0;
   	
	short r = 1;
	do{
		if(!s[idx1]){
			if(s[idx1+1]||s[idx1-1]||s[idx1+64]||s[idx1-64]){
				i1 = r;
			}else i1 = 0;
		}
		if(!s[idx2]){
			if(s[idx2+1]||s[idx2-1]||s[idx2+64]||s[idx2-64]){
				i2 = r;
			}else i2 = 0;		
		}
		if(!s[idx3]){
			if(s[idx3+1]||s[idx3-1]||s[idx3+64]||s[idx3-64]){
				s[idx3] = r;
			}else i3 = 0;		
		}
		if(!s[idx4]){
			if(s[idx4+1]||s[idx4-1]||s[idx4+64]||s[idx4-64]){
				s[idx4] = r;
			}else i4 = 0;		
		}
		__syncthreads();
		if(i1) s[idx1] = i1;
		if(i2) s[idx2] = i2;
		if(i3) s[idx3] = i3;
		if(i4) s[idx4] = i4;
		__syncthreads();
	}while(r++ < 21);
	
	/*for(py = -20; py <= 20; py++ ){
		for(px = -20; px <= 20; px++ ){
			if(s[iX-py+72*(iY+py)]){
				if(py*py+px*px < minDist)
					minDist = py*py+px*px;
			}
		}
	}*/
	if(!s[idx]){
		out[y*cols+x] = 21;
	}else if(s[idx] == 255){
		out[y*cols+x] = 0;
	}else
		out[y*cols+x] = s[idx];
}







__device__ void sumBM(unsigned char *p, unsigned char *s, unsigned int *r, unsigned int size ){
	
	int windowExt = size/2;
	int px = 15;
	int py = 15;
	int x = 0;
	int y = 0;
	r[threadIdx.x] = 0;// (int)p[py*31]*s[py*94+px-15]*(int)p[py*31]*s[py*94+px-15];
	
	for(x = -windowExt; x <= windowExt; x++){
		for(y = -windowExt; y <= windowExt; y++){
			r[threadIdx.x] += abs((int)p[(py+y)*31+(px+x)]-s[(py+y)*(31+64)+(px+x)+threadIdx.x]);
			//((int)p[(py+y)*size+(px+x)]-s[(py+y)*(size+64)+(px+x)+threadIdx.x])*((int)p[(py+y)*size+(px+x)]-s[(py+y)*(size+64)+(px+x)+threadIdx.x]);
		} 
	}
}
__device__ void findMinBM(unsigned int kernel[64], unsigned char *out, int* th){
	__shared__  unsigned char idx[32];
	out[1] = 0;
	if(threadIdx.x < 32){
		idx[threadIdx.x] = threadIdx.x;
	}
	__syncthreads();
	if(threadIdx.x < 32){
		if(kernel[threadIdx.x+32] < kernel[threadIdx.x]){
			kernel[threadIdx.x] = kernel[threadIdx.x+32];
			idx[threadIdx.x] = threadIdx.x+32;
		}else
			idx[threadIdx.x] = threadIdx.x;
	}
	__syncthreads();
	if(threadIdx.x < 16){
		if(kernel[threadIdx.x+16] < kernel[threadIdx.x]){
			kernel[threadIdx.x] = kernel[threadIdx.x+16];
			idx[threadIdx.x] = idx[threadIdx.x+16];
		}
	}
	__syncthreads();
	if(threadIdx.x < 8){
		if(kernel[threadIdx.x+8] < kernel[threadIdx.x]){
			kernel[threadIdx.x] = kernel[threadIdx.x+8];
			idx[threadIdx.x] = idx[threadIdx.x+8];
		}
	}
	__syncthreads();
	if(threadIdx.x < 4){
		if(kernel[threadIdx.x+4] < kernel[threadIdx.x]){
			kernel[threadIdx.x] = kernel[threadIdx.x+4];
			idx[threadIdx.x] = idx[threadIdx.x+4];
		}
	}
	__syncthreads();
	if(threadIdx.x < 2){
		if(kernel[threadIdx.x+2] < kernel[threadIdx.x]){
			kernel[threadIdx.x] = kernel[threadIdx.x+2];
			idx[threadIdx.x] = idx[threadIdx.x+2];
		}
	}
	__syncthreads();
	if(threadIdx.x < 1){
		if(kernel[threadIdx.x+1] < kernel[threadIdx.x]){
			kernel[threadIdx.x] = kernel[threadIdx.x+1];
			idx[threadIdx.x] = idx[threadIdx.x+1];
		}
		if(/*kernel[threadIdx.z] < *th && */idx[threadIdx.x] < 64)
			*out = idx[threadIdx.x];
		else
			*out = 0;
	}
}
__global__ void stereBM(const int rows, const int cols, unsigned char *left, unsigned char *right, unsigned char *edge, unsigned char *out, int windowSize){
	
	extern __shared__ unsigned char ext[];
	
	unsigned char* pattern = (unsigned char*)ext;
	unsigned char* search = (unsigned char*)&pattern[31*31];
	__shared__  unsigned int results[64];
	// Threads x 0-64
	// Threads y 16
	
	//windiw size 31
	//int windowExt = windowSize/2; //15
	int x = blockIdx.x * 1 + threadIdx.x;
	int y = blockIdx.y * 1 + threadIdx.y;
	
	//int px = threadIdx.x + windowExt;
	//int py = threadIdx.y + windowExt;
	
	/*if(threadIdx.x < 31){
		pattern[py*31+px-15] = 0; 
		if(threadIdx.y < 15){
			pattern[(py-15)*31+px-15] = 0; 
			pattern[(py+16)*31+px-15] = 0; 
		}
	}
	search[py*94+px-15] = 0; 
	if(threadIdx.y < 15){
		search[(py-15)*94+px-15] = 0; 
		search[(py+16)*94+px-15] = 0; 
	}
	if(threadIdx.x < 30){
		search[py*94+px-15+64] = 0; 
		if(threadIdx.y < 15){
			search[(py-15)*94+px-15+64] = 0; 
			search[(py+16)*94+px-15+64] = 0; 
		}
	}	*/
	__syncthreads();
	if(x < 15 || x >= cols-15-64 || y < 15 || y >= rows-15) return;
	if(threadIdx.x < 31){
		int iy = 0;
		for(iy = 0; iy < 31 ; iy++){
			pattern[iy*31+threadIdx.x] = left[(y-15+iy)*cols+(x-15)]; 
		}
	}
	int iy = 0;
	for(iy = 0; iy < 31 ; iy++){
		search[iy*94+threadIdx.x] = right[(y-15+iy)*cols+(x-15)]; 
	}
	if(threadIdx.x < 31){
		int iy = 0;
		for(iy = 0; iy < 31 ; iy++){
			search[iy*31+threadIdx.x+64] = right[(y-15+iy)*cols+(x-15+64)]; 
		}
	}
	
	/*
	if(threadIdx.x < 31){
		pattern[py*31+px-15] = left[(y)*cols+(x-15)]; 
		if(threadIdx.y < 15){
			pattern[(py-15)*31+px-15] = left[(y-15)*cols+(x-15)]; 
			pattern[(py+16)*31+px-15] = left[(y+16)*cols+(x-15)]; 
		}
	}
	search[py*94+px-15] = right[(y)*cols+(x-15)]; 
	if(threadIdx.y < 15){
		search[(py-15)*94+px-15] = right[(y-15)*cols+(x-15)]; 
		search[(py+16)*94+px-15] = right[(y+16)*cols+(x-15)]; 
	}
	if(threadIdx.x < 30){
		search[py*94+px-15+64] = right[(y)*cols+(x-15+64)]; 
		if(threadIdx.y < 15){
			search[(py-15)*94+px-15+64] = right[(y-15)*cols+(x-15+64)]; 
			search[(py+16)*94+px-15+64] = right[(y+16)*cols+(x-15+64)]; 
		}
	}*/
	__syncthreads();
	int eedge = (edge[y*cols+x]/2)+3;
	
	if(eedge > 31) return;
	sumBM(pattern, search, results, eedge);
	

	__syncthreads();
	unsigned char idx = 0;
	
	int th = 100000;
	findMinBM(results, &idx, &th);
	__syncthreads();
	if(threadIdx.x == 0)
		out[y*cols+x] = 4*idx;
		
}

__device__ void recFillEdge(unsigned char *edge, unsigned char *pattern, int x, int y, int px, int py, const int rows, const int cols, int depth){
	depth++;
	if(depth > 2) return;
	
	pattern[py*5+px] = 255;
	
	if(edge[(y)*(cols)+(x+1)] == 0 && pattern[py*5+px+1] == 0) recFillEdge(edge, pattern, x+1, y, px+1, py, rows, cols, depth);
	if(edge[(y)*(cols)+(x-1)] == 0 && pattern[py*5+px-1] == 0 ) recFillEdge(edge, pattern, x-1, y, px-1, py, rows, cols, depth);
	if(edge[(y+1)*(cols)+(x)] == 0 && pattern[(py+1)*5+px] == 0) recFillEdge(edge, pattern, x, y+1, px, py+1, rows, cols, depth);
	if(edge[(y-1)*(cols)+(x)] == 0 && pattern[(py-1)*5+px] == 0) recFillEdge(edge, pattern, x, y-1, px, py-1, rows, cols, depth);
	
	return;

}
__device__ void recFill7x7Edge(unsigned char *edge, unsigned char *pattern, int x, int y, int px, int py, const int rows, const int cols, int depth){
	depth++;
	if(depth > 3) return;
	
	pattern[py*7+px] = 255;
	
	if(edge[(y)*(cols)+(x+1)] == 0 && pattern[py*7+px+1] == 0) recFillEdge(edge, pattern, x+1, y, px+1, py, rows, cols, depth);
	if(edge[(y)*(cols)+(x-1)] == 0 && pattern[py*7+px-1] == 0) recFillEdge(edge, pattern, x-1, y, px-1, py, rows, cols, depth);
	if(edge[(y+1)*(cols)+(x)] == 0 && pattern[(py+1)*7+px] == 0) recFillEdge(edge, pattern, x, y+1, px, py+1, rows, cols, depth);
	if(edge[(y-1)*(cols)+(x)] == 0 && pattern[(py-1)*7+px] == 0) recFillEdge(edge, pattern, x, y-1, px, py-1, rows, cols, depth);
	
	return;

}
__global__ void median5x5Edge(const int rows, const int cols, unsigned char *edge, unsigned char *disp, unsigned char* out){
	
	unsigned char kernel[25];
	unsigned char pattern[25];
	unsigned char amount = 0;
	
	int x = blockIdx.x * blockDim.y + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	
	if(x < 3 || x >= cols-3 || y < 3 || y >= rows-3 ) return;
	
	if(edge[y*cols+x]) {
		out[y*cols+x] = disp[y*cols+x];
		return;
	}
	
	for(int ix = 0; ix < 5; ix++){
		for(int iy = 0; iy < 5; iy++){
			pattern[iy*5+ix] = 0;
		}
	} 
	kernel[0] = disp[y*cols+x];
	int r = 0;
	
	recFillEdge(edge, pattern, x, y, 2, 2, rows, cols, r);
	
	for(int ix = 0; ix < 5; ix++){
		for(int iy = 0; iy < 5; iy++){
			if(pattern[iy*5+ix]){
				amount++;
				kernel[amount] = disp[(y-2+iy)*cols+(x-2+ix)];
				
			}
		}
	}
	int tmp = 0;
	 for( int i = 0; i < amount-1; i++ )
		{
		    for( int j = 0; j < amount-2; j++ )
		    {
		        if( kernel[ j ] > kernel[ j + 1 ] ){
		        	tmp = kernel[j];
		        	kernel[j] = kernel[j+1];
		        	kernel[j+1] = tmp;
		        }
		    }
		}
	
	out[y*cols+x] = kernel[amount/2];
}


__global__ void blur5x5Edge(const int rows, const int cols, unsigned char *edge, unsigned char *disp, unsigned char* out){
	
	unsigned char pattern[25];
	unsigned char amount = 0;
	
	int x = blockIdx.x * blockDim.y + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	
	if(x < 3 || x >= cols-3 || y < 3 || y >= rows-3 ) return;
	
	if(edge[y*cols+x]) {
		out[y*cols+x] = disp[y*cols+x];
		return;
	}
	
	for(int ix = 0; ix < 7; ix++){
		for(int iy = 0; iy < 7; iy++){
			pattern[iy*7+ix] = 0;
		}
	} 
	int sum = 0;
	//kernel[0] = disp[y*cols+x];
	int r = 0;
	
	recFillEdge(edge, pattern, x, y, 3, 3, rows, cols, r);
	
	for(int ix = 0; ix < 7; ix++){
		for(int iy = 0; iy < 7; iy++){
			if(pattern[iy*7+ix]){
				amount++;
				sum += disp[(y-2+iy)*cols+(x-2+ix)];
			}
		}
	}
	
	out[y*cols+x] = sum/amount;
}



























