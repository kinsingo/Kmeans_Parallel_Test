#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
using namespace std;
using namespace cv;

class CL_Compute

{

private:

	cl::Context context;

	cl::Program program;

public:

	cl::Context Get_Context();

	cl::Program Get_Program();

private:

	cl::Program::Sources Get_Sources_From_kernel_cl_file_path(string kernel_cl_file_path);

	cl::Program Get_Program_From_kernel_cl_file_path(string kernel_cl_file_path);

public:

	CL_Compute(string kernel_cl_file_path);

	void Print_Platform_Info();

	cl::Device Get_default_device();

	cl::Buffer Get_ReadOnlyBuffer(cl::Context context, Mat mat);

	cl::Buffer Get_Buffer(cl::Context context, Mat mat);

};