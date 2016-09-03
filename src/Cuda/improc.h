


//-------------- CONVERSION --------------// 
__global__ void itoc_GPU(const int rows,const  int cols, int *img, unsigned char *result);
__global__ void ctoi_GPU(const int rows,const  int cols, unsigned char *img, int *result);


//--------------    BLUR    --------------//
__global__ void blur_GPU(const int rows,const  int cols, const int k, int *img, int *result);
//__global__ void blur_noShare_GPU( const int rows, const int cols,const int k, unsigned char *img, unsigned char *result);


//--------------    SOBEL   --------------//
__global__ void sobel_GPU (const int rows, const int cols, unsigned char *img, unsigned char *des, const int mode);
__global__ void sobel_abs_GPU (const int rows, const int cols, unsigned char *img, unsigned char *des, const int mode);


//--------------  PREWITT   --------------//
__global__ void prewitt_GPU (const int rows, const int cols, int *img, int *des, const int mode);
__global__ void prewittX_GPU (const int rows, const int cols, int *img, int *des, const int mode);
__global__ void prewittY_GPU (const int rows, const int cols, int *img, int *des, const int mode);
__global__ void prewittXsec_GPU (const int rows, const int cols, int *img, int *des, const int mode);
__global__ void prewittYsec_GPU (const int rows, const int cols, int *img, int *des, const int mode);
__global__ void prewittFS_GPU (const int rows, const int cols, int *img, int *desXf, int *desYf, int *desXs, int *desYs);


__global__ void edgeDetect(const int rows, const int cols, int *firstX, int *firstY, int *secX, int *secY,int *des);
__global__ void edgeDetect_GPU (const int rows, const int cols, unsigned char *img, unsigned char *out, int th);


//--------------  OTHER     --------------//
__global__ void blend_GPU (const int rows, const int cols, int *img1, int *img2, int *des, const float blend, const float scale);
__global__ void blend_GPU (const int rows, const int cols, unsigned char *img1, unsigned char *img2, unsigned char *des, const float blend, const float scale);
__global__ void rotate_GPU (const int rows, const int cols, unsigned char *img, unsigned char *des,float angle);



__global__ void edgeTypeDetect(const int rows, const int cols, unsigned char *img, unsigned char *des);

__global__ void findNode(const int rows, const int cols, unsigned char *img, unsigned char *des);
__global__ void extend(const int rows, const int cols, unsigned char *img, unsigned char *des);
__global__ void reduce(const int rows, const int cols, unsigned char *img, unsigned char *des);

__global__ void compare(const int rows, const int cols, unsigned char *imgL, unsigned char *imgR, unsigned char *des, int shift);



