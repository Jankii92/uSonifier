cmake_minimum_required(VERSION 2.8)

set(ProjectName DispToSonif)
project(${ProjectName})
set(Project_SRC src/main.cpp)
find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")

cuda_add_executable(${ProjectName} ${Project_SRC}) 

set(Csound1_LIBS libcsound64.so)
set(Csound2_LIBS libcsnd6.so)
set(DUO_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/lib/libDUO.so)

cuda_add_library( CameraCtrl "${CMAKE_CURRENT_SOURCE_DIR}/src/PID/cameractrl.cpp" )
cuda_add_library( Obstacle "${CMAKE_CURRENT_SOURCE_DIR}/src/Scene/Obstacle.cpp" )
cuda_add_library( Scene "${CMAKE_CURRENT_SOURCE_DIR}/src/Scene/Scene.cpp" )
cuda_add_library( CSoundManager "${CMAKE_CURRENT_SOURCE_DIR}/src/CSoundManager/CSoundManager.cpp" )
cuda_add_library( Cuda "${CMAKE_CURRENT_SOURCE_DIR}/src/Cuda/cuda.cu" )
cuda_add_library( CudaMJ "${CMAKE_CURRENT_SOURCE_DIR}/src/Cuda/cudamj.cpp" )
cuda_add_library( ImprocMJ "${CMAKE_CURRENT_SOURCE_DIR}/src/Cuda/improc.cu" )


target_link_libraries(${ProjectName} CameraCtrl CSoundManager Scene Obstacle CudaMJ Cuda ImprocMJ ${OpenCV_LIBS} ${DUO_LIBRARY} ${Csound1_LIBS} ${Csound2_LIBS}) 
add_definitions("-DUSE_DOUBLE")
