#include "../pch.h"
#include "cuda.h"

namespace cv{
	namespace gpu{
		void test();
		void blurmj(Mat &src, Mat &out, const int k);
		void sobelmj(Mat &src, Mat &out, const int mode);
		void dispmj(Mat &L, Mat &R, Mat &Out, unsigned char* l, unsigned char* r, unsigned char* disp, int shift);
		void realocHostMem(Mat &in);
		void setImageForCuda(Mat &mat, int size);
		void cpyImageForCuda(unsigned char* src, Mat &dest);
		void rectifmj(Mat &L, Mat &R, Mat &Out);
		void cudaInit(unsigned char** g_src1, unsigned char** g_src2, unsigned char** g_disp, const int rows, const int cols);
		void cudaDestroy(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp);

	
	}
}

	
