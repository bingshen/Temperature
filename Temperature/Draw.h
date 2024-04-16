#pragma once
#include "TemperatureMatrix.h"
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

class Draw
{
public:
	static void draw_pic(float pic[][PIC_SIZE],int file_no)
	{
        int size=PIC_SIZE;
        cv::Mat heatmap(size,size,CV_32FC1);
        for(int i=0;i<heatmap.rows;++i) 
        {
            for(int j=0;j<heatmap.cols;++j) 
            {
                heatmap.at<float>(i,j)=pic[i][j];
            }
        }
        cv::Mat normalizedHeatmap;
        cv::normalize(heatmap,normalizedHeatmap,0,255,cv::NORM_MINMAX,CV_8UC1);
        cv::Mat coloredHeatmap;
        cv::applyColorMap(normalizedHeatmap,coloredHeatmap,cv::COLORMAP_JET);
//        cv::imshow("Heatmap",coloredHeatmap);
//        cv::waitKey(0);
        string file_name=to_string(file_no);
        string file_path="pic/"+file_name+".png";
        cv::imwrite(file_path,coloredHeatmap);
	}
};