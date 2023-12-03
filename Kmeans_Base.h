#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <ctime>
#include <random>
#include <filesystem>
#include <omp.h>

using namespace cv;
using namespace std;

class Kmeans_Base
{
private:
    void Initialization(const Mat& data, int k, Mat& centers, Mat& cluster_indexes)
    {
        int num_points = data.rows;
        int num_dims = data.cols;
        RNG rng(1);
        vector<int> center_indices;
        for (int cluster = 0; cluster < k; ++cluster)
        {
            int idx = rng.uniform(0, num_points);
            center_indices.push_back(idx);
        }
        centers = Mat::zeros(k, num_dims, CV_32F);
        for (int cluster = 0; cluster < k; ++cluster)
            data.row(center_indices[cluster]).copyTo(centers.row(cluster));
        cluster_indexes = Mat::zeros(num_points, 1, CV_32SC1);
    }
protected:
    virtual void Conduct_Clustering(const Mat& data, const int k, Mat& centers, Mat& cluster_indexes, Mat& new_centers, Mat& cluster_counts) = 0;

public:
    void Kmeans(const Mat& data, int k, Mat& centers, Mat& cluster_indexes, int max_iter = 100, double epsillon = 1.0)
    {
        Initialization(data, k, centers, cluster_indexes);

        int num_dims = data.cols;
        for (int iter = 0; iter < max_iter; ++iter)
        {
            Mat new_centers = Mat::zeros(k, data.cols, CV_32F);
            Mat cluster_counts = Mat::zeros(k, 1, CV_32SC1);
            Conduct_Clustering(data, k, centers, cluster_indexes, new_centers, cluster_counts);

            double total_center_movement = 0.0;
            for (int cluster = 0; cluster < k; ++cluster)
            {
                if (cluster_counts.at<int>(cluster) > 0) {
                    new_centers.row(cluster) /= cluster_counts.at<int>(cluster);

                    double dist = 0.0;
                    for (int c = 0; c < num_dims; ++c) {
                        double diff = new_centers.at<float>(cluster, c) - centers.at<float>(cluster, c);
                        dist += diff * diff;
                    }
                    total_center_movement += dist;
                }
            }
            new_centers.copyTo(centers);

            double avg_center_movement = total_center_movement / k;
            if (avg_center_movement < epsillon)
                break;
        }
    }

    //void Kmeans(const Mat& data, int k, Mat& centers, Mat& cluster_indexes, int max_iter = 100, double epsillon = 1.0)
    //{
    //    Initialization(data, k, centers, cluster_indexes);
    //
    //    int num_points = data.rows;
    //    int num_dims = data.cols;
    //    for (int iter = 0; iter < max_iter; ++iter)
    //    {
    //        Mat new_centers = Mat::zeros(k, num_dims, CV_32F);
    //        Mat cluster_counts = Mat::zeros(k, 1, CV_32SC1);
    //        double total_center_movement = 0.0;
    //
    //
    //        Conduct_Clustering(data, k, centers, cluster_indexes, new_centers, cluster_counts);
    //
    //        for (int j = 0; j < k; ++j)
    //        {
    //            if (cluster_counts.at<int>(j) > 0) {
    //                new_centers.row(j) /= cluster_counts.at<int>(j);
    //                total_center_movement += squaredL2Distance(new_centers.row(j), centers.row(j));
    //            }
    //        }
    //        new_centers.copyTo(centers);
    //
    //        double avg_center_movement = total_center_movement / k;
    //        cout << "avg_center_movement:" << avg_center_movement << ", total_center_movement:" << total_center_movement << endl;
    //
    //        if (avg_center_movement < epsillon)
    //            break;
    //    }
    //}
};