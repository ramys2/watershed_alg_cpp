#pragma once

#include <future>
#include <SFML/Config.hpp>

#include "service/ImageLoader.hpp"
#include "service/ImageService.hpp"
#include "model/AppData.hpp"

class Controller final
{
private:
    ImageService mImageService;
    ImageLoader mImageLoader;
    AppData mAppData;
    std::future<cv::Mat> mTaskThread;

public:
    // Invokes service to load image
    void loadImage();
    // Invokes service to run manual implementation of watershed
    void runWatershedSegmentation();
    // Invokes service to run opencv implementation of watershed
    void runCvWatershedSegmentation();
    // Polls for result when segmentation is running
    void update();
    // Returns original texture
    const sf::Texture &retreiveOriginalImage() const;
};
