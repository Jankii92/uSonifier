
__device__ void cpyToShared16x16xR1(unsigned char* src, unsigned char out[18][18], int *x, int *y, const int rows, const int cols){
	
    out[threadIdx.x+1][threadIdx.y+1] = src[(*y)*cols+(*x)];
    
	if(threadIdx.x < 1){
    	if((*x) < 1){ 
    		out[threadIdx.x][threadIdx.y+1] = 0;
    		if(threadIdx.y < 1) out[threadIdx.x][threadIdx.y] = 0;
    	}else{
    		out[threadIdx.x][threadIdx.y+1] = src[(*y)*cols+(*x)-1];
    		if(*y < 1) out[threadIdx.x][threadIdx.y] = 0;
    		else out[threadIdx.x][threadIdx.y] = src[((*y)-1)*cols+((*x)-1)];  		
    	}
    }
    if(threadIdx.x+1 >= blockDim.x){
    	if(*x >= cols-1){ 
    		out[threadIdx.x+2][threadIdx.y+1] = 0;
    		if(threadIdx.y+1 >= blockDim.y) out[threadIdx.x+2][threadIdx.y+2] = 0;
    	}else{
    		out[threadIdx.x+2][threadIdx.y+1] = src[(*y)*cols+((*x)+1)];
    		if(*y >= rows-1) out[threadIdx.x+2][threadIdx.y+2] = 0;
    		else out[threadIdx.x+2][threadIdx.y+2] = src[((*y)+1)*cols+(*x)+1];
    	}
    }
    if(threadIdx.y < 1){
    	if(*y < 1){ 
    		out[threadIdx.x+1][threadIdx.y] = 0;
    		if(threadIdx.x+1 >= blockDim.x) out[threadIdx.x+2][threadIdx.y] = 0;
    	}else{
    		out[threadIdx.x+1][threadIdx.y] = src[((*y)-1)*cols+(*x)];
    		if(*x >= cols-1) out[threadIdx.x+2][threadIdx.y] = 0;
    		else out[threadIdx.x+2][threadIdx.y] = src[((*y)-1)*cols+(*x)+1];
    	}
    }
    if(threadIdx.y+1 >= blockDim.y){
    	if(*y >= rows-1){ 
    		out[threadIdx.x+1][threadIdx.y+2] = 0;
    		if(threadIdx.x+1 < 2) out[threadIdx.x][threadIdx.y+2] = 0;
    	}else{
    		out[threadIdx.x+1][threadIdx.y+2] = src[((*y)+1)*cols+(*x)];
    		if(*x < 1) out[threadIdx.x][threadIdx.y+2] = 0;
    		else out[threadIdx.x][threadIdx.y+2] = src[((*y)+1)*cols+((*x)-1)];
    	}
    }
}

__device__ void cpyToShared24x24xR1(unsigned char* src, unsigned char out[26][26], int *x, int *y, const int rows, const int cols){
	
    out[threadIdx.x+1][threadIdx.y+1] = src[(*y)*cols+(*x)];
    
	if(threadIdx.x < 1){
    	if(*x < 1){ 
    		out[threadIdx.x][threadIdx.y+1] = 0;
    		if(threadIdx.y < 1) out[threadIdx.x][threadIdx.y] = 0;
    	}else{
    		out[threadIdx.x][threadIdx.y+1] = src[(*y)*cols+(*x)-1];
    		if(*y < 1) out[threadIdx.x][threadIdx.y] = 0;
    		else out[threadIdx.x][threadIdx.y] = src[((*y)-1)*cols+((*x)-1)];  		
    	}
    }
    if(threadIdx.x+1 >= blockDim.x){
    	if(*x >= cols-1){ 
    		out[threadIdx.x+2][threadIdx.y+1] = 0;
    		if(threadIdx.y+1 >= blockDim.y) out[threadIdx.x+2][threadIdx.y+2] = 0;
    	}else{
    		out[threadIdx.x+2][threadIdx.y+1] = src[(*y)*cols+((*x)+1)];
    		if(*y >= rows-1) out[threadIdx.x+2][threadIdx.y+2] = 0;
    		else out[threadIdx.x+2][threadIdx.y+2] = src[((*y)+1)*cols+(*x)+1];
    	}
    }
    if(threadIdx.y < 1){
    	if(*y < 1){ 
    		out[threadIdx.x+1][threadIdx.y] = 0;
    		if(threadIdx.x+1 >= blockDim.x) out[threadIdx.x+2][threadIdx.y] = 0;
    	}else{
    		out[threadIdx.x+1][threadIdx.y] = src[((*y)-1)*cols+(*x)];
    		if(*x >= cols-1) out[threadIdx.x+2][threadIdx.y] = 0;
    		else out[threadIdx.x+2][threadIdx.y] = src[((*y)-1)*cols+(*x)+1];
    	}
    }
    if(threadIdx.y+1 >= blockDim.y){
    	if(*y >= rows-1){ 
    		out[threadIdx.x+1][threadIdx.y+2] = 0;
    		if(threadIdx.x+1 < 2) out[threadIdx.x][threadIdx.y+2] = 0;
    	}else{
    		out[threadIdx.x+1][threadIdx.y+2] = src[((*y)+1)*cols+(*x)];
    		if(*x < 1) out[threadIdx.x][threadIdx.y+2] = 0;
    		else out[threadIdx.x][threadIdx.y+2] = src[((*y)+1)*cols+((*x)-1)];
    	}
    }
}




