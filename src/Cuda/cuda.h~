

namespace cv{
	namespace gpu{
		namespace mj{	 
			void blur(const int rows,const int cols,const int k, unsigned char *src, unsigned char *dest);
			void sobel(const int rows,const int cols, unsigned char *src, unsigned char* dst, int mode);
			void rectif(const int rows,const int cols, unsigned char *srcL, unsigned char *srcR, unsigned char *destL, unsigned char *destR, unsigned char * out);
			void disp(const int rows,const int cols, unsigned char *srcL, unsigned char *srcR, unsigned char * out);
			void realocHostMem(int sizec, unsigned char *img);
			void cudaMemAlocImagePtr(unsigned char *dest, int size);
			void cudaMemcpyHtoH(unsigned char *src, unsigned char *dest, int size);
			void cudaMemcpyHtoD(unsigned char *src, unsigned char *dest, int size);
			void cudaMemcpyDtoH(unsigned char *src, unsigned char *dest, int size);
			void cudaInit(unsigned char** g_src1, unsigned char** g_src2, unsigned char** g_disp, const int rows, const int cols);
			void cudaDestroy(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp);
		}
	}	
}
