#include "pch.h"
#include "Cuda/cudamj.hpp"
#include "PID/cameractrl.h"
#include "CSoundManager/CSoundManager.h"
#include "duo.h"
#include <chrono>
#include <cuda_runtime.h>


#define FPS		30

using namespace cv::gpu;


int main(void){
	int counter = 0;
	int counter_sec = 0;
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp,0);
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
	
	cv::Mat disp( HEIGHT, WIDTH, CV_8UC1 );
	cv::Mat dispOut( HEIGHT, WIDTH, CV_8UC1 );
	
	cv::Mat dispOCV( HEIGHT, WIDTH, CV_8UC1 );
	
	//cv::Mat dispOCV( HEIGHT, WIDTH, CV_32F );
	
	setImageForCuda(left, sizeof(unsigned char)*WIDTH*HEIGHT);	
	setImageForCuda(right, sizeof(unsigned char)*WIDTH*HEIGHT);	
	setImageForCuda(leftOut, sizeof(unsigned char)*WIDTH*HEIGHT);	
	setImageForCuda(rightOut, sizeof(unsigned char)*WIDTH*HEIGHT);
	
	setImageForCuda(disp, sizeof(unsigned char)*WIDTH*HEIGHT);	
	setImageForCuda(dispOut, sizeof(unsigned char)*WIDTH*HEIGHT);
	
	unsigned char* g_imgL;
	unsigned char* g_imgR;
	unsigned char* g_disp;
	
	unsigned int* g_params;
	
	//unsigned char** g_tempsC = initDisp2C(sizeof(unsigned char)*WIDTH*HEIGHT);
	//unsigned int** g_tempsI = initDisp2I(sizeof(unsigned int)*WIDTH*HEIGHT);
	unsigned char** g_tempsC = initDisp3C(sizeof(unsigned char)*WIDTH*HEIGHT);
	unsigned short** g_tempsUS = initDisp3US(sizeof(unsigned int)*WIDTH*HEIGHT);
	
	//unsigned char** g_temps = initDisp(sizeof(unsigned char)*WIDTH*HEIGHT);
	cudaInit(&g_imgL, &g_imgR, &g_disp, HEIGHT, WIDTH);
	
	
	SetGain(0);
	SetExposure(20);
	SetLed(0);
	EnableUndistort();
	
	
	namedWindow( "DispCV", WINDOW_AUTOSIZE );
	gpu::GpuMat gL, gR, dst;
	gpu::StereoBM_GPU sbm(2, 64, 11);
	//gpu::DisparityBilateralFilter DBFilter(64 ,3, 2);
	Mat ocvDisp;
	
	
	CameraCtrl cameraCtrl( &left, SetExposure, SetGain, SetLed, false );
	
	//namedWindow( "Left", WINDOW_AUTOSIZE );
	//namedWindow( "Right", WINDOW_AUTOSIZE );
	namedWindow( "Disp", WINDOW_AUTOSIZE );
	namedWindow( "DispCV", WINDOW_AUTOSIZE );
	
	int shift = 0;
	auto start = std::chrono::system_clock::now();
	
	Mat cm_img0, image1, image2;
	image1 = imread("../data/im1.png");
	image2 = imread("../data/im0.png");
	
	cvtColor(image1(Rect(0,0,640,480)), image1, CV_BGR2GRAY);
	cvtColor(image2(Rect(0,0,640,480)), image2, CV_BGR2GRAY);
	//ocvDisp = imread("../data/disp0GT.bmp", 0);
	//ocvDisp.convertTo(ocvDisp,CV_8UC1); 
    //applyColorMap(4*ocvDisp, ocvDisp, COLORMAP_JET);
	while((cvWaitKey(5) & 0xff) != 27)
	{
		// Capture DUO frame
		
    	
		PDUOFrame pFrameData = GetDUOFrame();
		if(pFrameData == NULL) continue;
		
	//
		//cpyImageForCuda((unsigned char*)pFrameData->leftData, left);
		//cpyImageForCuda((unsigned char*)pFrameData->rightData, right);
		
		
		cpyImageForCuda((unsigned char*)(image1.data), left);
		cpyImageForCuda((unsigned char*)(image2.data), right);
		
		cameraCtrl.Update();
		
		auto start_blur = std::chrono::system_clock::now();

		//dispmj(left, right, disp, g_imgL, g_imgR, g_disp, g_temps);
		//disp2mj(left, right, disp, g_imgL, g_imgR, g_disp, g_tempsC, g_tempsI);
		disp3mj(left, right, disp, g_imgL, g_imgR, g_disp, g_tempsC, g_tempsUS);
		auto end_blur = std::chrono::system_clock::now();
		
		disp.convertTo(dispOut,CV_8UC1); 
		//bitwise_not ( dispOut, dispOut );
		//distanceTransform(dispOut, dispOut, CV_DIST_C, 3);
		//dispOut.convertTo(dispOut,CV_8UC1); 
        // Apply the colormap:
        
        applyColorMap(dispOut, dispOut, COLORMAP_JET);
		imshow("Disp", dispOut);
		//imshow("Left",left);
		//imshow("Right", right);
		
		/*gL.upload(left);
		gR.upload(right);
  		sbm(gR, gL, dst);
  		dst.download(ocvDisp);
  		ocvDisp.convertTo(ocvDisp,CV_8UC1); 
        applyColorMap(4*ocvDisp, ocvDisp, COLORMAP_JET);*/
	
		imshow("DispCV", left);
		counter++;
		auto stop = std::chrono::system_clock::now();
  		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  		
  		auto duration_blur = std::chrono::duration_cast<std::chrono::microseconds>(end_blur - start_blur);
  		
  		//if(shift>64) shift = 0;
		if(duration.count()>1000){
			//system("clear");	
			shift++;
			std::cout <<"FPS: "<< counter <<" "<< duration.count()/(float)counter <<  std::endl;
			std::cout <<"DISPARITY: "<< duration_blur.count()/1000.0f << "ms" << std::endl;
			counter = 0;
			start = std::chrono::system_clock::now();
		}
		
	}
	waitKey(0);
	//cudaDestroy(g_imgL, g_imgR, g_disp, g_temps);
	//cudaDestroyDisp2(g_imgL, g_imgR, g_disp, g_tempsC, g_tempsI);
	cudaDestroyDisp3(g_imgL, g_imgR, g_disp, g_tempsC, g_tempsUS);
	
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
