

namespace cv{
	namespace gpu{
		namespace mj{	 
			void blur(const int rows,const int cols,const int k, const unsigned char *src, unsigned char *dest);
			void sobel(const int rows,const int cols, const unsigned char *src, unsigned char* dst, int mode);
			void rectif(const int rows,const int cols, const unsigned char *srcL, const unsigned char *srcR, unsigned char *destL, unsigned char *destR, unsigned char * out);
			void realocHostMem(int sizec, unsigned char *img);
			void cudaMemAlocImagePtr(unsigned char *dest, int size);
			void cudaMemcpyHtoH(unsigned char *src, unsigned char *dest, int size);
		}
	}	
}
