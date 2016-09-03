#include "cudamj.hpp"
#include "chrono"

void cv::gpu::blurmj(Mat &src, Mat &out, const int k){
		auto start = std::chrono::system_clock::now();
		unsigned char* data = src.data;
		unsigned char* dataOut = out.data;
		
		auto stop1 = std::chrono::system_clock::now();
		cv::gpu::mj::blur(480, 640, k, data, dataOut);
		auto stop2 = std::chrono::system_clock::now();
		auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start);
		auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - stop1);
		//cout<<(int)duration1.count()<<" "<<(int)duration2.count()<< endl;
		
}

void cv::gpu::sobelmj(Mat &src, Mat &out, const int mode){
		unsigned char* data = src.data;
		unsigned char* dataOut = out.data;
		cv::gpu::mj::sobel(480, 640, data, dataOut, mode);
}


void cv::gpu::rectifmj(Mat &L, Mat &R, Mat &Out){
		unsigned char* dataL = L.data;
		unsigned char* dataR = R.data;
		unsigned char* data = Out.data;
		//unsigned char* dataOut = out.data;
		//cv::gpu::mj::disp(480, 640, dataL, dataR, data);
}

void cv::gpu::dispmj(Mat &L, Mat &R, Mat &Out, unsigned char* l, unsigned char* r, unsigned char* disp, int shift){
		unsigned char* dataL = L.data;
		unsigned char* dataR = R.data;
		unsigned char* data = Out.data;
		//unsigned char* dataOut = out.data;
		cv::gpu::mj::cudaMemcpyHtoD(dataL, l, 640*480*sizeof(unsigned char));
		cv::gpu::mj::cudaMemcpyHtoD(dataR, r, 640*480*sizeof(unsigned char));
		cv::gpu::mj::disp(480, 640, l, r, disp, shift);
		cv::gpu::mj::cudaMemcpyDtoH(disp, data, 640*480*sizeof(unsigned char));
}


			

void cv::gpu::realocHostMem(Mat &in){
	unsigned char* img =  in.data;
	int sizec =  (int)(sizeof(unsigned char)*in.size().height*in.size().width);
	cv::gpu::mj::realocHostMem(sizec, img);
	in.data = img;
}
void cv::gpu::setImageForCuda(Mat &mat, int size){
	unsigned char* matPtr = mat.data;
	cv::gpu::mj::cudaMemAlocImagePtr(matPtr, size);
}

void cv::gpu::cpyImageForCuda(unsigned char* src, Mat &dest){
	unsigned char* img = dest.data;
	int size = sizeof(unsigned char)*dest.size().height*dest.size().width;
	cv::gpu::mj::cudaMemcpyHtoH(src, img, size);
}

void cv::gpu::cudaInit(unsigned char** g_src1, unsigned char** g_src2, unsigned char** g_disp, const int rows, const int cols){
	cv::gpu::mj::cudaInit(g_src1, g_src2, g_disp, rows, cols);
}

void cv::gpu::cudaDestroy(unsigned char* g_src1, unsigned char* g_src2, unsigned char* g_disp){
	cv::gpu::mj::cudaDestroy(g_src1, g_src2, g_disp);
}



