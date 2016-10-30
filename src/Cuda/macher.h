
texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex2Dleft;
texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex2Dright;

cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<unsigned int>();
cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<unsigned int>();

__global__ void brain(const int rows, const int cols, unsigned char *left, unsigned char *right, unsigned char *edgeL, unsigned short* in8, unsigned short *weights, unsigned char *disp, int mode, int maxErr);
__global__ void udisp(const int rows, const int cols, unsigned char *disp, unsigned char *udisp);
__global__ void udispToUdepth(const int rows, const int cols, unsigned char *udisp, unsigned char *udepth);
