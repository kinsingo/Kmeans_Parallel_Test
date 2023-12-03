#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <ctime>
#include <random>
#include <filesystem>
#include <omp.h>
#include "Kmeans_Base.h"
#include "CL_Compute.h"

using namespace cv;
using namespace std;

class Kmeans_GPUPP : public Kmeans_Base
{
private:
    shared_ptr<CL_Compute> cl_Compute;
    cl::Kernel kernel;
    cl::CommandQueue queue;
public:
    Kmeans_GPUPP(shared_ptr<CL_Compute> cl_Compute)
    {
        this->cl_Compute = cl_Compute;
        kernel = cl::Kernel(cl_Compute->Get_Program(), "Kmeans");
        queue = cl::CommandQueue(cl_Compute->Get_Context());
    }

protected:
    void Conduct_Clustering(const Mat& data, const int k, Mat& centers, Mat& cluster_indexes, 
        Mat& new_centers, Mat& cluster_counts) override
    {
        int num_points = data.rows;
        int num_dims = data.cols;

        cl::Buffer dataBuffer(cl_Compute->Get_Context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * num_points * num_dims, data.data);
        cl::Buffer centersBuffer(cl_Compute->Get_Context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * k * num_dims, centers.data);

        cl::Buffer clusterIndexesBuffer(cl_Compute->Get_Context(), CL_MEM_WRITE_ONLY, sizeof(int) * num_points);
        cl::Buffer newCentersBuffer(cl_Compute->Get_Context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * k * num_dims, new_centers.data);
        cl::Buffer clusterCountsBuffer(cl_Compute->Get_Context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * k, cluster_counts.data);

        kernel.setArg(0, dataBuffer);
        kernel.setArg(1, centersBuffer);
        kernel.setArg(2, clusterIndexesBuffer);
        kernel.setArg(3, newCentersBuffer);
        kernel.setArg(4, clusterCountsBuffer);
        kernel.setArg(5, num_points);
        kernel.setArg(6, num_dims);
        kernel.setArg(7, k);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num_points), cl::NullRange);
        queue.finish();

        queue.enqueueReadBuffer(clusterIndexesBuffer, CL_TRUE, 0, sizeof(int) * num_points, cluster_indexes.data);
        queue.enqueueReadBuffer(newCentersBuffer, CL_TRUE, 0, sizeof(float) * k * num_dims, new_centers.data);
        queue.enqueueReadBuffer(clusterCountsBuffer, CL_TRUE, 0, sizeof(int) * k, cluster_counts.data);
    }
};


