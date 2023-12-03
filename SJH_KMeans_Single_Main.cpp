#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <ctime>
#include <random>
#include <filesystem>
#include <omp.h>
#include "Kmeans_Single.h"
#include "Kmeans_CPUPP.h"
#include "CL_Compute.h"
#include "Kmeans_GPUPP.h"
#include "Kmeans_CPUPP_Without_Critical.h"
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    // exePath를 기준으로 현재 디렉토리 설정
    std::filesystem::path exePath = std::filesystem::path(argv[0]).parent_path();
    std::filesystem::current_path(exePath);

    //CL object creation
    Kmeans_Single CPU_Single;
    Kmeans_CPUPP CPU_PP;
    Kmeans_CPUPP_Without_Critical CPU_PP_without_critical;
    shared_ptr<CL_Compute> cl_Compute = make_shared< CL_Compute>(std::filesystem::current_path().string() + "\\kernel_code.cl");
    cl_Compute->Print_Platform_Info();
    cl_Compute->Get_default_device();
    Kmeans_GPUPP GPU_PP(cl_Compute);

    // 이미지 파일을 읽어옵니다.
    Mat image = imread("fruits.jpg");
    cv::resize(image, image, cv::Size(3000, 3000));

    Mat OriginImage = image.clone();

    if (image.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return -1;
    }

    // 이미지를 2D 포인트 벡터로 변환합니다.
    Mat reshaped_image = image.reshape(1, image.rows * image.cols);//dims(2), rows(5797075), cols(3)
    Mat reshaped_image32f;//dims(2), rows(5797075), cols(3)
    reshaped_image.convertTo(reshaped_image32f, CV_32F);//dims(2), rows(5797075), cols(3)

    while (true)
    {
        int k;
        cout << "k:";
        cin >> k;

        int max_iter = 10;
        Mat cluster_indexes, centers;//dims(2), rows(5797075), cols(1) / dims(2), rows(50), cols(3)
        double epsillon = 0.1;//1.0; (기존은 1.0 이었음)
        cout << "image.rows :" << image.rows << ", image.cols :" << image.cols << ", k :" << k << ", epsillon : " << epsillon << endl;

        string Single_str = "Single,";
        string CPU_PP_str = "CPU_PP,";
        string CPU_PP_without_critical_str = "CPU_PP_without_critical,";
        string GPU_PP_str = "GPU_PP,";
        string OpenCV_PP_str = "OpenCV_PP,";
  
        for (int i = 0; i < 10; i++)
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            CPU_Single.Kmeans(reshaped_image32f, k, centers, cluster_indexes, max_iter);
            Single_str += to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count()) + ",";

            start_time = std::chrono::high_resolution_clock::now();
            CPU_PP.Kmeans(reshaped_image32f, k, centers, cluster_indexes, max_iter);
            CPU_PP_str += to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count()) + ",";

            start_time = std::chrono::high_resolution_clock::now();
            CPU_PP_without_critical.Kmeans(reshaped_image32f, k, centers, cluster_indexes, max_iter);
            CPU_PP_without_critical_str += to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count()) + ",";

            start_time = std::chrono::high_resolution_clock::now();
            GPU_PP.Kmeans(reshaped_image32f, k, centers, cluster_indexes, max_iter);
            GPU_PP_str += to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count()) + ",";

            TermCriteria termcrit(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);//최대 10번의 반복 또는 클러스터 중심의 이동 거리가 1.0 미만으로 감소하면 알고리즘을 종료합니다
            int attempts = 1;
            start_time = std::chrono::high_resolution_clock::now();
            kmeans(reshaped_image32f, k, cluster_indexes, termcrit, attempts, KMEANS_RANDOM_CENTERS, centers);
            OpenCV_PP_str += to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count()) + ",";
        }
        cout << Single_str << endl;
        cout << CPU_PP_str << endl;
        cout << CPU_PP_without_critical_str << endl;
        cout << GPU_PP_str << endl;
        cout << OpenCV_PP_str << endl;
    }
    cv::waitKey(0);

    return 0;
}



/*
int main(int argc, char* argv[])
{
    // exePath를 기준으로 현재 디렉토리 설정
    std::filesystem::path exePath = std::filesystem::path(argv[0]).parent_path();
    std::filesystem::current_path(exePath);

    //CL object creation
    Kmeans_Single CPU_Single;
    Kmeans_CPUPP CPU_PP;
    Kmeans_CPUPP_Without_Critical CPU_PP_without_critical;
    shared_ptr<CL_Compute> cl_Compute = make_shared< CL_Compute>(std::filesystem::current_path().string() + "\\kernel_code.cl");
    cl_Compute->Print_Platform_Info();
    cl_Compute->Get_default_device();
    Kmeans_GPUPP GPU_PP(cl_Compute);
    
    // 이미지 파일을 읽어옵니다.
    Mat image = imread("fruits.jpg");
    cv::resize(image, image, cv::Size(500, 500));
    Mat OriginImage = image.clone();

    if (image.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return -1;
    }

    // 이미지를 2D 포인트 벡터로 변환합니다.
    Mat reshaped_image = image.reshape(1, image.rows * image.cols);//dims(2), rows(5797075), cols(3)
    Mat reshaped_image32f;//dims(2), rows(5797075), cols(3)
    reshaped_image.convertTo(reshaped_image32f, CV_32F);//dims(2), rows(5797075), cols(3)

    int k = 50;
    int max_iter = 10;
    double epsillon = 1.0;
    cout << "image.rows :" << image.rows << ", image.cols :" << image.cols << ", k :" << k << ", epsillon : " << epsillon << endl;
    Mat cluster_indexes, centers;//dims(2), rows(5797075), cols(1) / dims(2), rows(50), cols(3)

    auto start_time = std::chrono::high_resolution_clock::now();
    CPU_Single.Kmeans(reshaped_image32f, k, centers, cluster_indexes, max_iter);
    std::cout << "K-Means (Single): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " milliseconds" << std::endl;
    
    start_time = std::chrono::high_resolution_clock::now();
    CPU_PP.Kmeans(reshaped_image32f, k, centers, cluster_indexes, max_iter);
    std::cout << "K-Means (CPU_PP): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " milliseconds" << std::endl;
    
    start_time = std::chrono::high_resolution_clock::now();
    CPU_PP_without_critical.Kmeans(reshaped_image32f, k, centers, cluster_indexes, max_iter);
    std::cout << "K-Means (CPU_PP_without_critical): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " milliseconds" << std::endl;
    
    start_time = std::chrono::high_resolution_clock::now();
    GPU_PP.Kmeans(reshaped_image32f, k, centers, cluster_indexes, max_iter);
    std::cout << "K-Means (GPU_PP): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " milliseconds" << std::endl;
    
    TermCriteria termcrit(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);//최대 10번의 반복 또는 클러스터 중심의 이동 거리가 1.0 미만으로 감소하면 알고리즘을 종료합니다
    int attempts = 1;
    start_time = std::chrono::high_resolution_clock::now();
    kmeans(reshaped_image32f, k, cluster_indexes, termcrit, attempts, KMEANS_RANDOM_CENTERS, centers);
    std::cout << "K-Means (OpenCV): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " milliseconds" << std::endl;

    //for (int i = 0; i < reshaped_image.rows; i++) 
    //{
    //    int cluster_idx = cluster_indexes.at<int>(i);
    //    Vec3b cluster_center = centers.at<Vec3f>(cluster_idx);
    //    reshaped_image.at<Vec3b>(i) = cluster_center;
    //}
    //
    //// 이미지를 원래 차원으로 변환합니다.
    //Mat clustered_image = reshaped_image.reshape(3, image.rows);
    //
    //// 클러스터링된 이미지를 표시합니다.
    //cv::resize(clustered_image, clustered_image, cv::Size(500, 500));
    //cv::resize(OriginImage, OriginImage, cv::Size(500, 500));
    //cv::Mat combined_image;
    //cv::hconcat(OriginImage, clustered_image, combined_image);
    //imshow("Left(Origin) Right(Clustered) Images", combined_image);
    cv::waitKey(0);

    return 0;
}
*/