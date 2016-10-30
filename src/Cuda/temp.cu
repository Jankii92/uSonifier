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
