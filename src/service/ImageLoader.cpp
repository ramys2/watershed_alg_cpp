#include "service/ImageLoader.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

cv::Mat ImageLoader::loadImage(const std::string& pathToImg) {
    return cv::imread(pathToImg, cv::IMREAD_GRAYSCALE);
}
