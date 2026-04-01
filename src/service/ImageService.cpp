#include "service/ImageService.hpp"

#include <iostream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

cv::Mat ImageService::watershedSegmentation(const cv::Mat matrix)
{
    std::cout << "Pre-processing image" << std::endl;
    cv::Mat preparedImg = preProcessForCustomWS(matrix);

    std::cout << "Returning result" << std::endl;
    return preparedImg;
}

cv::Mat ImageService::cvWatershedSegmentation(const cv::Mat matrix)
{
    throw std::logic_error("Not implemented yet!");
}

cv::Mat ImageService::preProcessForCustomWS(const cv::Mat &matrix)
{
    cv::Mat grayscale = convertToGreyScale(matrix);
    cv::GaussianBlur(grayscale, grayscale, cv::Size(7, 7), 0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(grayscale, grayscale, cv::MORPH_GRADIENT, kernel);

    return grayscale;
}

cv::Mat ImageService::convertToGreyScale(const cv::Mat &matrix)
{
    cv::Mat greyImg;
    cv::cvtColor(matrix, greyImg, cv::U8);
    return greyImg;
}

cv::Mat ImageService::postProcessingForCustomWS(const cv::Mat &matrix)
{
    cv::Mat rgbaImg;
    cv::cvtColor(matrix, rgbaImg, cv::COLOR_GRAY2RGBA);
    return rgbaImg;
}
