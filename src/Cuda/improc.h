

__global__ void blur_GPU(const int rows,const  int cols, const int k, const unsigned char *img, unsigned char *result);
__global__ void blur_noShare_GPU( const int rows, const int cols,const int k, const unsigned char *img, unsigned char *result);
