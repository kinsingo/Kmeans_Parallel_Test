#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <ctime>
#include <random>
#include <filesystem>
#include <omp.h>
#include "Kmeans_Base.h"

using namespace cv;
using namespace std;

class Kmeans_Single : public Kmeans_Base
{
protected:
    void Conduct_Clustering(const Mat& data, const int k, Mat& centers, 
        Mat& cluster_indexes, Mat& new_centers, Mat& cluster_counts) override
    {
        int num_points = data.rows;
        int num_dims = data.cols;
        for (int r = 0; r < num_points; ++r)
        {
            double min_dist = 999999;
            int closest_center = 0;
            for (int cluster = 0; cluster < k; ++cluster)
            {
                double dist = 0.0;
                for (int c = 0; c < num_dims; ++c){
                    double diff = data.at<float>(r, c) - centers.at<float>(cluster, c);
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    closest_center = cluster;
                }
            }

            cluster_indexes.at<int>(r) = closest_center;
            for (int c = 0; c < num_dims; ++c)
                new_centers.at<float>(closest_center, c) += data.at<float>(r, c);
            cluster_counts.at<int>(closest_center)++;
        }
    }
};


