

namespace cv{
	namespace gpu{
		namespace mj{	 
			void blur(const int rows,const int cols,const int k, unsigned char *src, unsigned char *dest);
			void sobel(const int rows,const int cols, unsigned char *src, unsigned char* dst, int mode);
			void rectif(const int rows,const int cols, unsigned char *srcL, unsigned char *srcR, unsigned char *destL, unsigned char *destR, unsigned char * out);
			void disp(const int rows,const int cols, unsigned char *srcL, unsigned char *srcR, unsigned char * out, unsigned char** temps);
			void realocHostMem(int sizec, unsigned char *img);
			void cudaMemAlocImagePtr(unsigned char *dest, int size);
			void cudaMemcpyHtoH(unsigned char *src, unsigned char *dest, int size);
			void cudaMemcpyHtoD(unsigned char *src, unsigned char *dest, int size);
			void cudaMemcpyDtoH(unsigned char *src, unsigned char *dest, int size);
			void cudaInit(unsigned char** g_src1, unsigned char** g_src2, unsigned char** g_disp, const int rows, const int cols);
			void cudaDestroy(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp, unsigned char** g_temps);
			void cudaDestroyDisp2(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp, unsigned char** g_tempsC, unsigned int** g_tempsI);
			void cudaDestroyDisp3(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp, unsigned char** g_tempsC, unsigned short** g_tempsUS);
			unsigned char** initDisp(const int size);
			unsigned int** initDisp2I(const int size);
			unsigned char** initDisp2C(const int size);
			unsigned short** initDisp3US(const int size);
			unsigned char** initDisp3C(const int size);
			void disp2(const int rows,const int cols, unsigned char *srcL, unsigned char *srcR, unsigned char * out, unsigned char** tempsC, unsigned int** tempsI);
			void disp3(const int rows,const int cols, unsigned char *srcL, unsigned char *srcR, unsigned char * out, unsigned char** tempsC, unsigned short** tempsUS);
			void dispToUdepth(const int rows,const int cols, const int uRows,const int uCols, unsigned char *g_disp, unsigned char *g_udepth, unsigned char** tempsC);

		}
	}	
}
