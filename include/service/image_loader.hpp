#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace image_loader
{
    // Loads new image and returns it's matrix
    cv::Mat loadImage(const std::string &pathToImg);
};
