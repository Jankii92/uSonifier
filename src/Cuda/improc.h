


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
__global__ void edgeDetect2x_GPU (const int rows, const int cols, unsigned char *img, unsigned char *out1, unsigned char *out2, int th1, int th2);


//--------------  OTHER     --------------//
__global__ void blend_GPU (const int rows, const int cols, int *img1, int *img2, int *des, const float blend, const float scale);
__global__ void blend_GPU (const int rows, const int cols, unsigned char *img1, unsigned char *img2, unsigned char *des, const float blend, const float scale);
__global__ void rotate_GPU (const int rows, const int cols, unsigned char *img, unsigned char *des,float angle);



__global__ void edgeTypeDetect(const int rows, const int cols, unsigned char *img, unsigned char *des);
__global__ void edgeTypeDetectCleanup(const int rows, const int cols, unsigned char *img, unsigned char *des);

__global__ void findNode(const int rows, const int cols, unsigned char *img, unsigned char *des);
__global__ void extend(const int rows, const int cols, unsigned char *img, unsigned char *des);
__global__ void reduce(const int rows, const int cols, unsigned char *img, unsigned char *des);

__global__ void compare(const int rows, const int cols, unsigned char *imgL, unsigned char *imgR, unsigned char *des, int shift);

__global__ void extender(const int rows, const int cols, unsigned char *low, unsigned char *high, unsigned char *edge);


__global__ void difference(const int rows, const int cols, unsigned char *in1, unsigned char *in2, unsigned char *dif);
__global__ void inprove(const int rows, const int cols, unsigned char *in1, unsigned char *in2, unsigned char *out);

__global__ void edgeMacher(const int rows, const int cols, unsigned char *left, unsigned char *right, unsigned char *srcL, unsigned char *srcR, unsigned char *disp);
__global__ void edgeEstimate(const int rows, const int cols, unsigned char *left, unsigned char *leftLow,  unsigned char *disp, unsigned char *dispOut,const int mode);

__global__ void edgeMatch2(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY ) ;
__global__ void edgeMatch8(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY) ;
__global__ void edgeMatch4(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR,  unsigned int *out, const int shiftX,  const int shiftY) ;
__global__ void edgeMatch12(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY) ;
__global__ void edgeMatch16(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY) ;
__global__ void edgeMatch32(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned int *out, const int shiftX,  const int shiftY) ;


__global__ void brain(const int rows, const int cols, unsigned char *edgeL, int *dis4, int *dis8, int *dis12, int *dis16, unsigned char *out);
__global__ void brain1(const int rows, const int cols, unsigned char *edgeL, unsigned int* i0, unsigned int* i1,unsigned int* i2,unsigned int* i3,unsigned int* i4, unsigned int* i0x, unsigned int* i1x,unsigned int* i2x,unsigned int* i3x,unsigned int* i4x, unsigned int* i0y, unsigned int* i1y,unsigned int* i2y,unsigned int* i3y,unsigned int* i4y, unsigned int* i0xy, unsigned int* i1xy,unsigned int* i2xy,unsigned int* i3xy,unsigned int* i4xy, unsigned int *weights,  unsigned char *disp);
__global__ void brain2(const int rows, const int cols, unsigned char *edgeL, unsigned short* in8, unsigned short *weights, unsigned char *disp);
__global__ void brain3(const int rows, const int cols, unsigned char *left, unsigned char *right, unsigned char *edgeL, unsigned short* in8, unsigned short *weights, unsigned char *disp);
__global__ void brain11(const int rows, const int cols, unsigned char *edgeL, unsigned short* i8, unsigned short *weights, unsigned char *disp);


__global__ void median(const int rows, const int cols, unsigned char *src, unsigned char *edge, unsigned char *med);
__global__ void findDistance(const int rows, const int cols, unsigned char *edge, unsigned char *out);
__global__ void findDistanceFast(const int rows, const int cols, unsigned char *edge, unsigned char *out);
__global__ void euclidian_distance_transform_kernel(unsigned char* img, unsigned char* dist, int w, int h);

__global__ void kernelDT(short* output, int size, float threshold2, short xm, short ym, short xM, short yM);


__global__ void stereBM(const int rows, const int cols, unsigned char *left, unsigned char *right, unsigned char *edge, unsigned char *out, int widndowSize);
__global__ void median5x5Edge(const int rows, const int cols, unsigned char *edge, unsigned char *disp, unsigned char* out);
__global__ void blur5x5Edge(const int rows, const int cols, unsigned char *edge, unsigned char *disp, unsigned char* out);


__global__ void edgeMatch8w16(const int rows, const int cols, unsigned char *edgeL, unsigned char *edgeR, unsigned short *out);




