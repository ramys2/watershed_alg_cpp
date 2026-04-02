#include "service/ImageLoader.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

cv::Mat ImageLoader::loadImage(const std::string &pathToImg)
{
    cv::Mat loadedImage = cv::imread(pathToImg, cv::IMREAD_COLOR);

    if (loadedImage.empty())
    {
        return cv::Mat();
    }

    cv::Mat rgbaImage;

    cv::cvtColor(loadedImage, rgbaImage, cv::COLOR_BGR2RGBA);

    return rgbaImage;
}
