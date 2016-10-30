cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(ProjectName DispToSonif)
project(${ProjectName})
set(Project_SRC src/main.cpp)


SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})

find_package(ZED REQUIRED)


find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


link_directories(${ZED_LIBRARY_DIR})
include_directories(${ZED_INCLUDE_DIRS})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -use_fast_math -arch sm_32)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11 -Wl -pedantic)
cuda_add_executable(${ProjectName} ${Project_SRC}) 

set(Csound1_LIBS libcsound64.so)
set(Csound2_LIBS libcsnd6.so)
set(DUO_LIBRARY ${CMAKE_SOURCE_DIR}/lib/libDUO.so)

cuda_add_library( CameraCtrl "${CMAKE_SOURCE_DIR}/src/PID/cameractrl.cpp" )
cuda_add_library( Obstacle "${CMAKE_SOURCE_DIR}/src/Scene/Obstacle.cpp" )
cuda_add_library( Scene "${CMAKE_SOURCE_DIR}/src/Scene/Scene.cpp" )
cuda_add_library( CSoundManager "${CMAKE_SOURCE_DIR}/src/CSoundManager/CSoundManager.cpp" )
cuda_add_library( Cuda "${CMAKE_SOURCE_DIR}/src/Cuda/cuda.cu" )
cuda_add_library( CudaMJ "${CMAKE_SOURCE_DIR}/src/Cuda/cudamj.cpp" )
cuda_add_library( ImprocMJ "${CMAKE_SOURCE_DIR}/src/Cuda/improc.cu" )
cuda_add_library( MacherMJ "${CMAKE_SOURCE_DIR}/src/Cuda/macher.cu" )


target_link_libraries(${ProjectName} CameraCtrl CSoundManager Scene Obstacle CudaMJ Cuda ImprocMJ MacherMJ  ${OpenCV_LIBS} ${ZED_LIBRARIES} ${DUO_LIBRARY} ${Csound1_LIBS} ${Csound2_LIBS} ${PCL_LIBRARIES}) 
add_definitions("-DUSE_DOUBLE")
