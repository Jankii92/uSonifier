

__global__ void blur_GPU(const int rows,const  int cols, const int k, const unsigned char *img, unsigned char *result);
__global__ void blur_noShare_GPU( const int rows, const int cols,const int k, const unsigned char *img, unsigned char *result);
__global__ void rectify_GPU (const int rows, const int cols, const unsigned char *imgL,const unsigned char *imgR, int *rot);
__global__ void sobel_GPU (const int rows, const int cols, const unsigned char *img, unsigned char *des, const int mode);
__global__ void sobel_abs_GPU (const int rows, const int cols, const unsigned char *img, unsigned char *des, const int mode);


