

#define DISP_MIN 0
#define DISP_MAX 63
__device__ unsigned int __usad4(unsigned int A, unsigned int B, unsigned int C=0)
{
    unsigned int result;
#if (__CUDA_ARCH__ >= 300) // Kepler (SM 3.x) supports a 4 vector SAD SIMD
    asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;": "=r"(result):"r"(A), "r"(B), "r"(C));
#else // SM 2.0            // Fermi  (SM 2.x) supports only 1 SAD SIMD, so there are 4 instructions
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b0, %2.b0, %3;": "=r"(result):"r"(A), "r"(B), "r"(C));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b1, %2.b1, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b2, %2.b2, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b3, %2.b3, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
#endif
    return result;
}



__device__ unsigned short absus(short a)
{
  return max(-a, a);
}
__device__ void match8x64w16(unsigned char pattern[16][16], unsigned char search[64+16][16], unsigned short* result, const unsigned short* idx){
	unsigned short* result2 = (unsigned short*)&result[1024];
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

__device__ void sum8x64w16(unsigned short *result, const unsigned short* idx){
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
	
	
	const int  x = blockIdx.x * 16 + threadIdx.x;
	const int  y = blockIdx.y * blockDim.y + threadIdx.y;
	const int  t = y*cols+x;
	const unsigned short idx = threadIdx.y*64+threadIdx.x;
	
    //result[threadIdx.y*64+threadIdx.x] = 0; possible setting max uchar
    if(x < 0 || x >= cols-64-16 || y < 0 || y >= rows){
			if(threadIdx.y == 0){
			out[(2*blockIdx.y*(80)+2*blockIdx.x)*64+threadIdx.x] = 60000;
			out[(2*blockIdx.y*(80)+(2*blockIdx.x+1))*64+threadIdx.x] = 60000;
		}
		if(threadIdx.y == 8){
			out[((2*(blockIdx.y)+1)*(80)+2*blockIdx.x)*64+threadIdx.x] = 60000;
			out[((2*(blockIdx.y)+1)*(80)+2*blockIdx.x+1)*64+threadIdx.x] = 60000;
		}
    return;
    }
    
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
__device__ void match2extend_16x16x2(unsigned char pattern[18][18], unsigned char search[64+18][18], unsigned short *block2, const unsigned short shift){
	
	/*short iX = threadIdx.x+1;
	short iY = threadIdx.y+1;
	int idx = ((threadIdx.y * 16 + threadIdx.x)*64) + threadIdx.z + shift;
	short p[9];
	 
	
	/*block2[idx] =    absus((short)pattern[iX-1][iY-1] - search[iX-1+shift+threadIdx.z][iY-1]);///// Optimalisation
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
	block2[idx+6] +=   absus((short)pattern[iX+1][iY+1] - search[iX+1+shift+threadIdx.z+6][iY+1]);*/
	
	/*p[0] = (short)pattern[iX-1][iY-1];
	block2[idx] =    absus(p[0] - search[iX-1+shift+threadIdx.z][iY-1]);
	block2[idx+2] =    absus(p[0] - search[iX-1+shift+threadIdx.z+2][iY-1]);
	block2[idx+4] =    absus(p[0] - search[iX-1+shift+threadIdx.z+4][iY-1]);
	block2[idx+4] =    absus(p[0] - search[iX-1+shift+threadIdx.z+4][iY-1]);
	block2[idx+6] =    absus(p[0] - search[iX-1+shift+threadIdx.z+6][iY-1]);
	
	p[1] = (short)pattern[iX  ][iY-1];
	block2[idx] += 2*absus(p[1] - search[iX  +shift+threadIdx.z][iY-1]);
	block2[idx+2] += 2*absus(p[1] - search[iX  +shift+threadIdx.z+2][iY-1]);
	block2[idx+4] += 2*absus(p[1] - search[iX  +shift+threadIdx.z+4][iY-1]);
	block2[idx+6] += 2*absus(p[1] - search[iX  +shift+threadIdx.z+6][iY-1]);
	
	p[2] = (short)pattern[iX+1][iY-1];
	block2[idx] +=   absus(p[2] - search[iX+1+shift+threadIdx.z][iY-1]);
	block2[idx+2] +=   absus(p[2] - search[iX+1+shift+threadIdx.z+2][iY-1]);
	block2[idx+4] +=   absus(p[2] - search[iX+1+shift+threadIdx.z+4][iY-1]);
	block2[idx+6] +=   absus(p[2] - search[iX+1+shift+threadIdx.z+6][iY-1]);
	
	p[3] = (short)pattern[iX-1][iY  ];
	block2[idx] += 2*absus(p[3] - search[iX-1+shift+threadIdx.z][iY  ]);
	block2[idx+2] += 2*absus(p[3] - search[iX-1+shift+threadIdx.z+2][iY  ]);
	block2[idx+4] += 2*absus(p[3] - search[iX-1+shift+threadIdx.z+4][iY  ]);
	block2[idx+6] += 2*absus(p[3] - search[iX-1+shift+threadIdx.z+6][iY  ]);
	
	p[4] = (short)pattern[iX  ][iY  ];
	block2[idx] += 4*absus(p[4] - search[iX  +shift+threadIdx.z][iY  ]);
	block2[idx+2] += 4*absus(p[4] - search[iX  +shift+threadIdx.z+2][iY  ]);
	block2[idx+4] += 4*absus(p[4] - search[iX  +shift+threadIdx.z+4][iY  ]);
	block2[idx+6] += 4*absus(p[4] - search[iX  +shift+threadIdx.z+6][iY  ]);
	
	p[5] = (short)pattern[iX+1][iY  ];
	block2[idx] += 2*absus(p[5] - search[iX+1+shift+threadIdx.z][iY  ]);
	block2[idx+2] += 2*absus(p[5] - search[iX+1+shift+threadIdx.z+2][iY  ]);
	block2[idx+4] += 2*absus(p[5] - search[iX+1+shift+threadIdx.z+4][iY  ]);
	block2[idx+6] += 2*absus(p[5] - search[iX+1+shift+threadIdx.z+6][iY  ]);
	
	p[6] = (short)pattern[iX-1][iY+1];
	block2[idx] +=   absus(p[6] - search[iX-1+shift+threadIdx.z][iY+1]);
	block2[idx+2] +=   absus(p[6] - search[iX-1+shift+threadIdx.z+2][iY+1]);
	block2[idx+4] +=   absus(p[6] - search[iX-1+shift+threadIdx.z+4][iY+1]);
	block2[idx+6] +=   absus(p[6] - search[iX-1+shift+threadIdx.z+6][iY+1]);
	
	p[7] = (short)pattern[iX  ][iY+1];
	block2[idx] += 2*absus(p[7] - search[iX  +shift+threadIdx.z][iY+1]);
	block2[idx+2] += 2*absus(p[7] - search[iX  +shift+threadIdx.z+2][iY+1]);
	block2[idx+4] += 2*absus(p[7] - search[iX  +shift+threadIdx.z+4][iY+1]);
	block2[idx+6] += 2*absus(p[7] - search[iX  +shift+threadIdx.z+6][iY+1]);
	
	p[8] = (short)pattern[iX-1][iY+1];
	block2[idx] +=   absus(p[8] - search[iX+1+shift+threadIdx.z][iY+1]);
	block2[idx+2] +=   absus(p[8] - search[iX+1+shift+threadIdx.z+2][iY+1]);
	block2[idx+4] +=   absus(p[8] - search[iX+1+shift+threadIdx.z+4][iY+1]);
	block2[idx+6] +=   absus(p[8] - search[iX+1+shift+threadIdx.z+6][iY+1]);*/
	
	const short iX = shift+threadIdx.z+1;
	const short iY = threadIdx.y+1;
	const int idx = ((threadIdx.y * 16 + (shift+threadIdx.z))*64) + threadIdx.x;
	
	short p[9];
	p[0] = (short)pattern[iX-1][iY-1];
	p[1] = (short)pattern[iX  ][iY-1];
	p[2] = (short)pattern[iX+1][iY-1];
	p[3] = (short)pattern[iX-1][iY  ];
	p[4] = (short)pattern[iX  ][iY  ];
	p[5] = (short)pattern[iX+1][iY  ];
	p[6] = (short)pattern[iX-1][iY+1];
	p[7] = (short)pattern[iX  ][iY+1];
	p[8] = (short)pattern[iX+1][iY+1];
	
	block2[idx]    = absus(p[0] - search[iX+threadIdx.x-1 ][threadIdx.y]);
	block2[idx+16] = absus(p[0] - search[iX+threadIdx.x+15][threadIdx.y]);
	block2[idx+32] = absus(p[0] - search[iX+threadIdx.x+31][threadIdx.y]);
	block2[idx+48] = absus(p[0] - search[iX+threadIdx.x+47][threadIdx.y]);
	
	block2[idx] +=    absus(p[1] - search[iX+threadIdx.x   ][threadIdx.y]);
	block2[idx+16] += absus(p[1] - search[iX+threadIdx.x+16][threadIdx.y]);
	block2[idx+32] += absus(p[1] - search[iX+threadIdx.x+32][threadIdx.y]);
	block2[idx+48] += absus(p[1] - search[iX+threadIdx.x+48][threadIdx.y]);
	
	block2[idx] +=    absus(p[2] - search[iX+threadIdx.x+1 ][threadIdx.y]);
	block2[idx+16] += absus(p[2] - search[iX+threadIdx.x+17][threadIdx.y]);
	block2[idx+32] += absus(p[2] - search[iX+threadIdx.x+33][threadIdx.y]);
	block2[idx+48] += absus(p[2] - search[iX+threadIdx.x+49][threadIdx.y]);
	
	block2[idx] +=    absus(p[3] - search[iX+threadIdx.x-1 ][iY]);
	block2[idx+16] += absus(p[3] - search[iX+threadIdx.x+15][iY]);
	block2[idx+32] += absus(p[3] - search[iX+threadIdx.x+31][iY]);
	block2[idx+48] += absus(p[3] - search[iX+threadIdx.x+47][iY]);
	
	block2[idx] +=    absus(p[4] - search[iX+threadIdx.x   ][iY]);
	block2[idx+16] += absus(p[4] - search[iX+threadIdx.x+16][iY]);
	block2[idx+32] += absus(p[4] - search[iX+threadIdx.x+32][iY]);
	block2[idx+48] += absus(p[4] - search[iX+threadIdx.x+48][iY]);
	
	block2[idx] +=    absus(p[5] - search[iX+threadIdx.x+1 ][iY]);
	block2[idx+16] += absus(p[5] - search[iX+threadIdx.x+17][iY]);
	block2[idx+32] += absus(p[5] - search[iX+threadIdx.x+33][iY]);
	block2[idx+48] += absus(p[5] - search[iX+threadIdx.x+49][iY]);
	
	block2[idx] +=    absus(p[6] - search[iX+threadIdx.x-1 ][iY+1]);
	block2[idx+16] += absus(p[6] - search[iX+threadIdx.x+15][iY+1]);
	block2[idx+32] += absus(p[6] - search[iX+threadIdx.x+31][iY+1]);
	block2[idx+48] += absus(p[6] - search[iX+threadIdx.x+47][iY+1]);
	
	block2[idx] +=    absus(p[7] - search[iX+threadIdx.x   ][iY+1]);
	block2[idx+16] += absus(p[7] - search[iX+threadIdx.x+16][iY+1]);
	block2[idx+32] += absus(p[7] - search[iX+threadIdx.x+32][iY+1]);
	block2[idx+48] += absus(p[7] - search[iX+threadIdx.x+48][iY+1]);
	
	block2[idx] +=    absus(p[8] - search[iX+threadIdx.x+1 ][iY+1]);
	block2[idx+16] += absus(p[8] - search[iX+threadIdx.x+17][iY+1]);
	block2[idx+32] += absus(p[8] - search[iX+threadIdx.x+33][iY+1]);
	block2[idx+48] += absus(p[8] - search[iX+threadIdx.x+49][iY+1]);
	
}
__device__ int roundff(float a)
{
  return (int)floor(a + 0.5);
}
__device__ void findBestDispXX(unsigned int *in, unsigned char *indexes, unsigned int *minTemp){

	const int inIdx = threadIdx.x + threadIdx.z*16; // after optimalisation
	const int idxRes = threadIdx.y * 64 + inIdx;
	const int idxIdx = threadIdx.y * 32 + inIdx;
	int tmpIdx = 0;
	int tmpVal = 0;
	float interMinX =0.0f;
		
	if(in[idxRes+32] < in[idxRes]){
			minTemp[idxIdx] = in[idxRes+32];
			indexes[idxIdx] = inIdx+32;
		}else{
			indexes[idxIdx] = inIdx;
			minTemp[idxIdx] = in[idxRes];
		}
	__syncthreads();
	if(inIdx < 16){
		if(minTemp[idxIdx+16] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+16];
			indexes[idxIdx] = indexes[idxIdx+16];
		}
	}
	__syncthreads();
	if(inIdx < 8){
		if(minTemp[idxIdx+8] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+8];
			indexes[idxIdx] = indexes[idxIdx+8];
		}
	}
	__syncthreads();
	if(inIdx < 4){
		if(minTemp[idxIdx+4] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+4];
			indexes[idxIdx] = indexes[idxIdx+4];
		}
	}
	__syncthreads();
	if(inIdx < 2){
		if(minTemp[idxIdx+2] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+2];
			indexes[idxIdx] = indexes[idxIdx+2];
		}
	}
	__syncthreads();
	if(inIdx == 0){
		if(minTemp[idxIdx+1] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+1];
			indexes[idxIdx] = indexes[idxIdx+1];
		}
		//in[idxRes] = minTemp[idxIdx];
		tmpVal = in[threadIdx.y * 64+indexes[idxIdx]];
		tmpIdx = indexes[idxIdx];
		if(tmpIdx != 0 && tmpIdx != 63 ){
			interMinX = 4*(float)((int)in[threadIdx.y * 64+tmpIdx-1]-(int)in[threadIdx.y * 64+tmpIdx+1])/(2*(((int)in[threadIdx.y * 64+tmpIdx-1])-2*(int)in[threadIdx.y * 64+tmpIdx]+(int)in[threadIdx.y * 64+tmpIdx+1]));
		}
		in[threadIdx.y * 64+tmpIdx] = 99999999;
		
		/*if(indexes[idxIdx] == 0){
			in[threadIdx.y * 64+tmpIdx+1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+2] = 99999999;
			in[threadIdx.y * 64+tmpIdx+3] = 99999999;
			in[threadIdx.y * 64+tmpIdx+4] = 99999999;
		}
		else if(indexes[idxIdx] == 1){
			in[threadIdx.y * 64+tmpIdx-1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+2] = 99999999;
			in[threadIdx.y * 64+tmpIdx+3] = 99999999;
			in[threadIdx.y * 64+tmpIdx+4] = 99999999;
		}		
		else if(indexes[idxIdx] == 2){
			in[threadIdx.y * 64+tmpIdx-2] = 99999999;
			in[threadIdx.y * 64+tmpIdx-1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+2] = 99999999;
			in[threadIdx.y * 64+tmpIdx+3] = 99999999;
			in[threadIdx.y * 64+tmpIdx+4] = 99999999;
		}
		else if(indexes[idxIdx] == 61){
			in[threadIdx.y * 64+tmpIdx-4] = 99999999;
			in[threadIdx.y * 64+tmpIdx-3] = 99999999;
			in[threadIdx.y * 64+tmpIdx-2] = 99999999;
			in[threadIdx.y * 64+tmpIdx-1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+2] = 99999999;
		}
		else if(indexes[idxIdx] == 62){
			in[threadIdx.y * 64+tmpIdx-4] = 99999999;
			in[threadIdx.y * 64+tmpIdx-3] = 99999999;
			in[threadIdx.y * 64+tmpIdx-2] = 99999999;
			in[threadIdx.y * 64+tmpIdx-1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+1] = 99999999;
		}
		else if(indexes[idxIdx] == 63){
			in[threadIdx.y * 64+tmpIdx-4] = 99999999;
			in[threadIdx.y * 64+tmpIdx-3] = 99999999;
			in[threadIdx.y * 64+tmpIdx-2] = 99999999;
			in[threadIdx.y * 64+tmpIdx-1] = 99999999;
		}else{
			in[threadIdx.y * 64+tmpIdx-4] = 99999999;
			in[threadIdx.y * 64+tmpIdx-3] = 99999999;
			in[threadIdx.y * 64+tmpIdx-2] = 99999999;
			in[threadIdx.y * 64+tmpIdx-1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+2] = 99999999;
			in[threadIdx.y * 64+tmpIdx+3] = 99999999;
			in[threadIdx.y * 64+tmpIdx+4] = 99999999;
		}*/
		if(indexes[idxIdx] == 0){
			in[threadIdx.y * 64+tmpIdx+1] = 99999999;
		}
		else if(indexes[idxIdx] == 63){
			in[threadIdx.y * 64+tmpIdx-1] = 99999999;
		}else{
			in[threadIdx.y * 64+tmpIdx-1] = 99999999;
			in[threadIdx.y * 64+tmpIdx+1] = 99999999;
		}
	}
	__syncthreads();
	if(in[idxRes+32] < in[idxRes]){
			minTemp[idxIdx] = in[idxRes+32];
			indexes[idxIdx] = inIdx+32;
		}else{
			indexes[idxIdx] = inIdx;
			minTemp[idxIdx] = in[idxRes];
		}
	__syncthreads();
	if(inIdx < 16){
		if(minTemp[idxIdx+16] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+16];
			indexes[idxIdx] = indexes[idxIdx+16];
		}
	}
	__syncthreads();
	if(inIdx < 8){
		if(minTemp[idxIdx+8] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+8];
			indexes[idxIdx] = indexes[idxIdx+8];
		}
	}
	__syncthreads();
	if(inIdx < 4){
		if(minTemp[idxIdx+4] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+4];
			indexes[idxIdx] = indexes[idxIdx+4];
		}
	}
	__syncthreads();
	if(inIdx < 2){
		if(minTemp[idxIdx+2] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+2];
			indexes[idxIdx] = indexes[idxIdx+2];
		}
	}
	__syncthreads();
	if(inIdx == 0){
		if(minTemp[idxIdx+1] < minTemp[idxIdx]){
			minTemp[idxIdx] = minTemp[idxIdx+1];
			indexes[idxIdx] = indexes[idxIdx+1];
		}
		//in[idxRes] = minTemp[idxIdx];
			if(abs(tmpIdx-indexes[idxIdx]) > 4){
				if(0.6f > (float)(in[threadIdx.y * 64+indexes[idxIdx]]-tmpVal)/(tmpVal)){//*tmpIdx)){ // 0.02
					indexes[idxIdx] = 0;
					indexes[idxIdx+1] = 4;
				}else{
					indexes[idxIdx] = tmpIdx;
					indexes[idxIdx+1] = (int)(round(interMinX))+4;
				in[threadIdx.y * 64] = tmpVal;
				}		
			}else if(abs(tmpIdx-indexes[idxIdx]) > 1){
				if(0.15f > (float)(in[threadIdx.y * 64+indexes[idxIdx]]-tmpVal)/(tmpVal)){//*tmpIdx)){ // 0.02
					indexes[idxIdx] = 0;
					indexes[idxIdx+1] = 4;
				}else{
					indexes[idxIdx] = tmpIdx;
					indexes[idxIdx+1] = (int)(round(interMinX))+4;
				in[threadIdx.y * 64] = tmpVal;
				}	
			}else{
				indexes[idxIdx] = tmpIdx;
				indexes[idxIdx+1] = (int)(round(interMinX))+4;
				in[threadIdx.y * 64] = tmpVal;
			}
	}
}
__global__ void brain3(const int rows, const int cols, unsigned char *left, unsigned char *right, unsigned char *edgeL, unsigned short* in8, unsigned short *weights, unsigned char *disp, int mode, int maxErr){
	__shared__  unsigned char pattern[18][18];
	__shared__  unsigned char search[64+18][18];
	__shared__ unsigned short w[32];
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int t = y*cols+x;
	
	
	extern __shared__ unsigned int extt[];
	
	unsigned int *block16 = (unsigned int*)&extt[0];
	//size of block 16 is 2x2x64
	
	unsigned short *block8 = (unsigned short*)&block16[3*3*64];
	//size of block8 is 6x6x64
	
	unsigned short *block2 = (unsigned short*)&block8[6*6*64];
	//size of block2 is 16x16x64
	
	unsigned int *res = (unsigned int*)&block2[16*16*64];
	
	unsigned char *indexes = (unsigned char*)&res[4*4*64];
	
	unsigned int *minTemp = (unsigned int*)&indexes[4*4*32];
	
	if(threadIdx.y == 0){
		w[threadIdx.x+16*threadIdx.z] = weights[threadIdx.x+16*threadIdx.z];
	}

	if(blockIdx.x < 1 || blockIdx.x >= (cols/16)-1 || blockIdx.y < 1 || blockIdx.y >= (rows/16)-1) return; 
//------fill pattern---------//
	const int iX = threadIdx.x+1;
	const int iY = threadIdx.y+1;
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
	if(mode){
		match2extend_16x16x2(pattern, search, block2, 0);
		match2extend_16x16x2(pattern, search, block2, 2);
		match2extend_16x16x2(pattern, search, block2, 4);
		match2extend_16x16x2(pattern, search, block2, 6);
		match2extend_16x16x2(pattern, search, block2, 8);
		match2extend_16x16x2(pattern, search, block2, 10);
		match2extend_16x16x2(pattern, search, block2, 12);
		match2extend_16x16x2(pattern, search, block2, 14);
	}

//----copy edgeDistanceTransform to pattern-----------//
	if(threadIdx.z == 0)
		search[threadIdx.x][threadIdx.y] = edgeL[t];

//------calculate 16x16 blocks----------//
	const int b16idx = ((threadIdx.y/4)*3+(threadIdx.x/4))*64;
	const int b8idx  = ((threadIdx.y/4)*6+(threadIdx.x/4))*64*2;
	const int zidx2 = 2*(4*(threadIdx.y%4) + (threadIdx.x%4))+threadIdx.z;
	
	if(threadIdx.x < 12 && threadIdx.y < 12){
		block16[b16idx+zidx2]   = block8[b8idx+zidx2];
		block16[b16idx+zidx2+32] = block8[b8idx+zidx2+32];
		
		block16[b16idx+zidx2]   += block8[b8idx+zidx2+64];
		block16[b16idx+zidx2+32] += block8[b8idx+zidx2+32+64];
		
		block16[b16idx+zidx2]   += block8[b8idx+zidx2+64*6];
		block16[b16idx+zidx2+32] += block8[b8idx+zidx2+32+64*6];
		
		block16[b16idx+zidx2]   += block8[b8idx+zidx2+64*7];
		block16[b16idx+zidx2+32] += block8[b8idx+zidx2+32+64*7];
		
	}
	__syncthreads();
	
//----calculate(3x3) 16x16 blocks into (3x) 32x32 blocks with results in corners

	const int zidx = (2*(4*(threadIdx.y%8)+(threadIdx.x%4)))+threadIdx.z;
	
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
	const int shift = 16*threadIdx.z + threadIdx.x; 
	const int idxRes = threadIdx.y*64 + shift;

	const int idxB32_1 = 0*64+shift;
	const int idxB32_2 = 2*64+shift;
	const int idxB32_3 = 6*64+shift;
	const int idxB32_4 = 8*64+shift;
	const int idxB16_1 = 3*64+shift;
	const int idxB16_2 = 1*64+shift;
	const int idxB16_3 = 7*64+shift;
	const int idxB16_4 = 5*64+shift;

	if(blockIdx.x < 1 || blockIdx.x >= (cols/16)-4 || blockIdx.y < 1 || blockIdx.y >= (rows/16)-1) return;

	int ix, iy, ixx, iyy;
	
	const int thYmod = threadIdx.y%4;
	const int thYdiv = threadIdx.y/4;
	float xx32;// for next iter 0 change to 4/8/12
	float yy32;// for next iter 0 change to 4/8/12
	float xx16;// for next iter 0 change to 4/8/12
	float yy16;// for next iter 0 change to 4/8/12
	float x32;
	float y32;
	float x16;
	float y16;
	unsigned short idx2;
	unsigned short weight;
	
	#pragma unroll
	for(ix = 0 ; ix < 2 ; ix++){
		#pragma unroll
		for(iy = 0 ; iy < 2; iy++){
			if(threadIdx.x < 4 && threadIdx.y < 8){
				block16[1*64+zidx] =  block8[(8+6*iy+ix)*64+zidx];
				block16[1*64+zidx] += block8[(9+6*iy+ix)*64+zidx];
				block16[1*64+zidx] += block8[(14+6*iy+ix)*64+zidx];
				block16[1*64+zidx] += block8[(15+6*iy+ix)*64+zidx];
			}
			if(threadIdx.x < 4 && threadIdx.y >= 8 && threadIdx.y < 16 ){
				block16[3*64+zidx] =  block8[(7+6*iy+ix)*64+zidx];
				block16[3*64+zidx] += block8[(8+6*iy+ix)*64+zidx];
				block16[3*64+zidx] += block8[(13+6*iy+ix)*64+zidx];
				block16[3*64+zidx] += block8[(14+6*iy+ix)*64+zidx];
			}
			if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y < 8){
				block16[64*5+zidx] =  block8[64*(14+6*iy+ix)+zidx];
				block16[64*5+zidx] += block8[64*(15+6*iy+ix)+zidx];
				block16[64*5+zidx] += block8[64*(20+6*iy+ix)+zidx];
				block16[64*5+zidx] += block8[64*(21+6*iy+ix)+zidx];
			}
			if(threadIdx.x >= 4 && threadIdx.x < 8 && threadIdx.y >= 8 && threadIdx.y < 16 ){
				block16[64*7+zidx] =  block8[64*(13+6*iy+ix)+zidx];
				block16[64*7+zidx] += block8[64*(14+6*iy+ix)+zidx];
				block16[64*7+zidx] += block8[64*(19+6*iy+ix)+zidx];
				block16[64*7+zidx] += block8[64*(20+6*iy+ix)+zidx];
			}
			
			__syncthreads();
			#pragma unroll
			for(ixx = 8*ix ; ixx < 8+8*ix ; ixx+=4){
				#pragma unroll
				for(iyy = 8*iy ; iyy < 8+8*iy; iyy+=4){
					xx32 = ((float)(ixx+(thYmod)))/16.0;// for next iter 0 change to 4/8/12
					yy32 = ((float)(iyy+(thYdiv)))/16.0;// for next iter 0 change to 4/8/12
					xx16 = ((float)(ixx-8*ix+(thYmod)))/8.0;// for next iter 0 change to 4/8/12
					yy16 = ((float)(iyy-8*iy+(thYdiv)))/8.0;// for next iter 0 change to 4/8/12
					x32  = 1-xx32;
					y32  = 1-yy32;
					x16  = 1-xx16;
					y16  = 1-yy16;
					res[idxRes] = (x32*y32*block16[idxB32_1])+(xx32*y32*block16[idxB32_2])+(x32*yy32*block16[idxB32_3])+(xx32*yy32*block16[idxB32_4]);
					res[idxRes+32] = (x32*y32*block16[idxB32_1+32])+(xx32*y32*block16[idxB32_2+32])+(x32*yy32*block16[idxB32_3+32])+(xx32*yy32*block16[idxB32_4+32]);
	
					res[idxRes] += 4*((x16*y16*block16[idxB16_1])+(xx16*y16*block16[idxB16_2])+(x16*yy16*block16[idxB16_3])+(xx16*yy16*block16[idxB16_4]));
					res[idxRes+32] += 4*((x16*y16*block16[idxB16_1+32])+(xx16*y16*block16[idxB16_2+32])+(x16*yy16*block16[idxB16_3+32])+(xx16*yy16*block16[idxB16_4+32]));
					if(mode){
						idx2 = ((iyy+(thYdiv))*16+(ixx+(thYmod)))*64+shift;
						weight = w[search[ixx+(thYmod)][iyy+(thYdiv)]];
						res[idxRes] += weight*block2[idx2];
						res[idxRes+32] += weight*block2[idx2+32];
					}
					__syncthreads();
				//------find max of sums -------------//
					findBestDispXX(res, indexes, minTemp);
					__syncthreads();	
				//------save best results into the file-----//
					if(threadIdx.x >= ixx && threadIdx.x < ixx+4 && threadIdx.y >= iyy && threadIdx.y < iyy+4 && threadIdx.z == 0){
						if(res[(4*(threadIdx.y%4)+(threadIdx.x%4))*64] < maxErr && pattern[threadIdx.x+1][threadIdx.y+1] < 255)
							disp[t] = 4*(int)(indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32])+(((int)indexes[(4*(threadIdx.y%4)+(threadIdx.x%4))*32+1])-4);
						else
							disp[t] = 0;
					}
					__syncthreads();
				}
			}
		}
	}
}	
__global__ void udisp(const int rows, const int cols, unsigned char *disp, unsigned char *udisp){
		
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
		
	udisp[disp[y*cols+x]*cols+x]++;	
}

__global__ void udispToUdepth(const int rows, const int cols, unsigned char *udisp, unsigned char *udepth){
		
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
		
    const double v0 = 327.9689445495605;
	if(y < 2) return;
	float Zw = 10.0f*4.0f/(y);
	float ZwNext = 10.0f*4.0f/(y-1);
	if(ZwNext > 4.0f) return;
	
    float Xw = (float) ((x - v0) * Zw / 333.333);
	int Z = 480-(int)roundff(Zw*100);
	int Znext = 480-(int)roundff(ZwNext*100);
	int X = 320+round(Xw*100);
	if(Z >= 0 && Z < 480 && X >= 0 && X < 640){
		for(int yi = Znext+1; yi <= Z; yi++)
			udepth[yi*cols+X] = udisp[y*cols+x];
	}
}
	


