#include "pch.h"
#include "Cuda/cudamj.hpp"
#include "PID/cameractrl.h"
#include "CSoundManager/CSoundManager.h"
#include "duo.h"
#include <chrono>


#define FPS		30

using namespace cv::gpu;


int main(void){
	int counter = 0;
	int counter_sec = 0;
	// Open DUO camera and start capturing
	printf("DUOLib Version:       v%s\n", GetLibVersion());
	if(!OpenDUOCamera(WIDTH, HEIGHT, FPS))
	{
		printf("Could not open DUO camera\n");
		return 0;
	}
	printf("\nHit <ESC> to exit.\n");
	
	cv::Mat left(  HEIGHT, WIDTH, CV_8UC1 );
	cv::Mat leftOut(  HEIGHT, WIDTH, CV_8UC1 );
	cv::Mat right( HEIGHT, WIDTH, CV_8UC1 );
	cv::Mat rightOut( HEIGHT, WIDTH, CV_8UC1 );
	
	setImageForCuda(left, sizeof(unsigned char)*WIDTH*HEIGHT);	
	setImageForCuda(right, sizeof(unsigned char)*WIDTH*HEIGHT);	
	setImageForCuda(leftOut, sizeof(unsigned char)*WIDTH*HEIGHT);	
	setImageForCuda(rightOut, sizeof(unsigned char)*WIDTH*HEIGHT);
	
	SetGain(0);
	SetExposure(20);
	SetLed(0);
	EnableUndistort();
	//StereoBM sbm;
	CameraCtrl cameraCtrl( &left, SetExposure, SetGain, SetLed, false );
	
	namedWindow( "Left", WINDOW_AUTOSIZE );
	namedWindow( "Right", WINDOW_AUTOSIZE );
	auto start = std::chrono::system_clock::now();
	while((cvWaitKey(1) & 0xff) != 27)
	{
		// Capture DUO frame
		
    	
		PDUOFrame pFrameData = GetDUOFrame();
		if(pFrameData == NULL) continue;
		cpyImageForCuda((unsigned char*)pFrameData->leftData, left);
		cpyImageForCuda((unsigned char*)pFrameData->rightData, right);
		
		cameraCtrl.Update();
		
		auto start_blur = std::chrono::system_clock::now();
   		blurmj(left, leftOut, 5);
   		blurmj(right, rightOut, 5);
   		//blur(left, leftOut, Size(11,11));
   		//blur(right, rightOut, Size(11,11));
		auto end_blur = std::chrono::system_clock::now();
		imshow("Left", leftOut);
		imshow("Right", rightOut);
  		
		counter++;
		auto stop = std::chrono::system_clock::now();
  		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  		
  		auto duration_blur = std::chrono::duration_cast<std::chrono::microseconds>(end_blur - start_blur);
		if(duration.count()>1000){
			//system("clear");	
			std::cout <<"FPS: "<< counter <<" "<< duration.count()/(float)counter <<  std::endl;
			std::cout <<"BLUR: "<< duration_blur.count()/1000.0f << "ms" << std::endl;
			counter = 0;
			start = std::chrono::system_clock::now();
		}
		
	}
	
	
	/*
	Mat image;
	
	realocHostMem(image);
    auto start1 = std::chrono::system_clock::now();
	realocHostMem(image);
	image = imread("../data/Lenna.png", 0);
	auto stop1 = std::chrono::system_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
	//cvtColor(image, image, CV_RGB2GRAY);
	namedWindow( "Original", WINDOW_AUTOSIZE );
	namedWindow( "CPU", WINDOW_AUTOSIZE );
	namedWindow( "GPU", WINDOW_AUTOSIZE );
	namedWindow( "GPU2", WINDOW_AUTOSIZE );
    Mat imageGPU = image.clone();
    Mat imageCPU = image.clone();
    Mat imageCPU2;
    
    
    blurmj(image, imageGPU, 51);
	
	
	auto start2 = std::chrono::system_clock::now();
    blur(image, imageCPU, Size(51,51));
	auto stop2 = std::chrono::system_clock::now();
	auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    
    GpuMat gpumatOut;
    GpuMat gpumatIn(image);
    cv::gpu::blur(gpumatIn, gpumatOut, Size(21,21));
    auto start3 = std::chrono::system_clock::now();
    cv::gpu::blur(gpumatIn, gpumatOut, Size(21,21));
    gpumatOut.download(imageCPU2);
	auto stop3 = std::chrono::system_clock::now();
	auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3);
    
    
    cout<<(int)duration1.count()<<endl;
    cout<<(int)duration2.count()<<endl;
    cout<<(int)duration3.count()<<endl;
    //test();
    
    
    imshow( "Original", image );  
    imshow( "GPU", imageGPU);
    imshow( "GPU2", imageCPU2);
    imshow( "CPU", imageCPU);
    */
	//char* csdfile = "/home/ubuntu/Documents/DUO3D-ARM-v1.0.50.26/DUOSDK/Samples/OpenCV/Sample-01-cvShowImage/csound.csd";
	//Scene *scene = new Scene(SCENE_MIN_DEPTH, SCENE_MAX_DEPTH);
	//CSoundManager *cs = new CSoundManager(scene, csdfile);
	//cs->Start();
	//cs->Stop();
	
	CloseDUOCamera();
	
	waitKey(100);
	return 0;
}