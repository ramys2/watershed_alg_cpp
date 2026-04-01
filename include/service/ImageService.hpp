#pragma once

#include <opencv2/core.hpp>

class ImageService final
{
public:
    cv::Mat watershedSegmentation(const cv::Mat matrix);
    cv::Mat cvWatershedSegmentation(const cv::Mat matrix);

private:
    cv::Mat preProcessForCustomWS(const cv::Mat &matrix);
    cv::Mat convertToGreyScale(const cv::Mat &matrix);
    cv::Mat postProcessingForCustomWS(const cv::Mat &matrix);
};
