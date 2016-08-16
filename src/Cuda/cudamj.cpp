#include "cudamj.hpp"
#include "chrono"


void cv::gpu::test(){
	cv::gpu::mj::test();
}
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


