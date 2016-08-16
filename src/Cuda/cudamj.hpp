#include "../pch.h"
#include "cuda.h"

namespace cv{
	namespace gpu{
		void test();
		void blurmj(Mat &src, Mat &out, const int k);
		void realocHostMem(Mat &in);
		void setImageForCuda(Mat &mat, int size);
		void cpyImageForCuda(unsigned char* src, Mat &dest);
	
	}
}

	
