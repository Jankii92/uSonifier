#include "improc.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>

#define BLUR_K 5

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

__global__ void prewittXY_GPU (const int rows, const int cols, int *img, int *des, const int mode){
		
		//const int r = 1;
		
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x <= 1 || x >= cols-1 || y <= 1 || y >= rows-1) return;
		
    	
    	int sumXY = 0;
    	    	
    	sumXY-=2*img[(y-1)*cols+(x-1)];
    	sumXY+=img[(y-1)*cols+(x)];
    	//sumX+=img[(y-1)*cols+(x+1)];
    	sumXY-=img[(y)*cols+(x-1)];
    	//sum+=img[(y)*cols+(x)];
    	sumXY+=img[(y)*cols+(x+1)]; 
    	//sumX-=img[(y+1)*cols+(x-1)];
    	sumXY+=img[(y+1)*cols+(x)];
    	sumXY+=2*img[(y+1)*cols+(x+1)];
    	
		if(mode == 0) 
			des[y*cols+x] = (sumXY);
}

__global__ void prewittYX_GPU (const int rows, const int cols, int *img, int *des, const int mode){
		
		//const int r = 1;
		
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x <= 1 || x >= cols-1 || y <= 1 || y >= rows-1) return;
		
    	
    	int sumYX = 0;
    	    	
    	//sumYX-=img[(y-1)*cols+(x-1)];
    	sumYX+=img[(y-1)*cols+(x)];
    	sumYX+=2*img[(y-1)*cols+(x+1)];
    	
    	sumYX-=img[(y)*cols+(x-1)];
    	//sumYX+=img[(y)*cols+(x-1)];
    	sumYX+=img[(y)*cols+(x+1)]; 
    	
    	sumYX-=2*img[(y+1)*cols+(x-1)];
    	sumYX-=img[(y+1)*cols+(x)];
    	//sumYX+=img[(y+1)*cols+(x+1)];

		if(mode == 0) 
			des[y*cols+x] = (sumYX);
}

__global__ void prewittX_GPU (const int rows, const int cols, int *img, int *des, const int mode){
		
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x <= 1 || x >= cols-1 || y <= 1 || y >= rows-1) return;

    	int sumX = 0;
    	    	
    	sumX-=img[(y-1)*cols+(x-1)];
    	//sum+=((int)img[(y-1)*cols+(x)]*sX[1];
    	sumX+=img[(y-1)*cols+(x+1)];
    	sumX-=2*img[(y)*cols+(x-1)];
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[4];
    	sumX+=2*img[(y)*cols+(x+1)]; 
    	sumX-=img[(y+1)*cols+(x-1)];
    	//sum+=((int)img[(y+1)*cols+(x)]*sX[7];
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
    	    	
    	/*//sumX-=img[(y-1)*cols+(x-1)];
    	//sum+=((int)img[(y-1)*cols+(x)]*sX[1];
    	//sumX+=img[(y-1)*cols+(x+1)];
    	sumX+=1*img[(y)*cols+(x-1)];
    	sumX-=2*img[(y)*cols+(x)];
    	sumX+=1*img[(y)*cols+(x+1)]; 
    	//sumX-=img[(y+1)*cols+(x-1)];
    	//sum+=((int)img[(y+1)*cols+(x)]*sX[7];
    	//sumX+=img[(y+1)*cols+(x+1)];*/
    	
    	
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
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[3];
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[4];
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[5];
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
    		
    	//if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
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

    	__shared__ unsigned char s[21][21];
    	
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
    	if(iX >= blockDim.x-1){
    		if(x >= cols-3) s[iX+3][iY] = 0;
    		else{
    			s[iX+3][iY] = img[y*cols+(x+3)];
    			if(y >= rows-3) s[iX+3][iY+3] = 0;
    			else s[iX+3][iY+3] = img[(y+3)*cols+x+3];
    		}
    	}
    	if(iY < 4){
    		if(y < 2) s[iX][iY-2] = 0;
    		else{
    			s[iX][iY-2] = img[(y-2)*cols+x];
    			if(x >= cols-3) s[iX+3][iY-2] = 0;
    			else s[iX+3][iY-2] = img[(y-2)*cols+x+3];
    		}
    	}
    	if(iY >= blockDim.y-1){
    		if(y >= rows-3) s[iX][iY+3] = 0;
    		else{
    			s[iX][iY+3] = img[(y+3)*cols+x];
    			if(x < 2) s[iX-2][iY+3] = 0;
    			else s[iX-2][iY+3] = img[(y+3)*cols+(x-2)];
    		}
    	}
    	
    	__syncthreads();
    		
    	//if(x < 0 || x >= cols || y < 0 || y >= rows) return;
    	
    	int sumYf = 0;
    	int sumYs = 0; 
    	int sumYss = 0;   	
    	int sumXf = 0;
    	int sumXs = 0;
    	int sumXss = 0;
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
		
		
		if(sumXf > th || sumXf < -th){
			if(!SameSign_GPU(&sumXs, &sumXss))
				out[y*cols+x] = 255;
				return;
		}		
		else if(sumYf > th || sumYf < -th){
			if(!SameSign_GPU(&sumYs, &sumYss))
				out[y*cols+x] = 255;
				return;
		}
		out[y*cols+x] = 0;
		
}

__global__ void prewittYsec_GPU (const int rows, const int cols, int *img, int *des, const int mode){
		
		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x <= 2 || x >= cols-2 || y <= 2 || y >= rows-2) return;
    	
    	int sumY = 0;
    	/*
    	//sumY-=img[(y-1)*cols+(x-1)];
    	sumY+=1*img[(y-1)*cols+(x)];
    	//sumY-=img[(y-1)*cols+(x+1)];
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[3];
    	sumY-=2*img[(y)*cols+(x)];
    	//sum+=((int)img[(y)*cols+(x-1)]*sX[5];
    	//sumY+=img[(y+1)*cols+(x-1)];
    	sumY+=1*img[(y+1)*cols+(x)];
    	//sumY+=img[(y+1)*cols+(x+1)];*/
    	
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

__global__ void edgeDetect(const int rows, const int cols, int *firstX, int *firstY, int *secX, int *secY, int *des){

		int x = blockIdx.x * blockDim.x + threadIdx.x;
    	int y = blockIdx.y * blockDim.y + threadIdx.y;

    	if(x < 1 || x > cols-1 || y < 1 || y > rows-1) return;
		
		int firstTH = 30;
		//int secTH = 30;
		if(abs(firstX[y*cols+x]) > firstTH ){
			des[y*cols+x] = 0;
			if(!SameSign(secX[y*cols+(x-1)], secX[y*cols+(x)]))
				des[y*cols+x] = 255;
				return;
		}		
		else if(abs(firstY[y*cols+x]) > firstTH ){
			des[y*cols+x] = 0;
			if(!SameSign(secY[(y-1)*cols+(x)], secY[(y)*cols+(x)]))
				des[y*cols+x] = 255;
				return;
		}		

		else 
			des[y*cols+x] = 0;
}










