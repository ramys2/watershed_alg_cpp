#include "service/ImageService.hpp"

#include <stdexcept>
#include <opencv2/imgproc.hpp>

cv::Mat ImageService::watershedSegmentation(const cv::Mat &matrix)
{
    throw std::logic_error("Not implemented yet!");
}

cv::Mat ImageService::cvWatershedSegmentation(const cv::Mat &matrix)
{
    throw std::logic_error("Not implemented yet!");
}

cv::Mat ImageService::convertToGreyScale(const cv::Mat &matrix)
{
    cv::Mat greyImg;
    cv::cvtColor(matrix, greyImg, cv::COLOR_RGBA2GRAY);
    return greyImg;
}
