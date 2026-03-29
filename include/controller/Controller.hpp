#pragma once

#include <string>
#include <future>

#include "service/ImageLoader.hpp"
#include "service/ImageService.hpp"
#include "model/AppData.hpp"

class Controller final {
    private:
        ImageService mImageService;
        ImageLoader mImageLoader;
        AppData mAppData;
        std::future<cv::Mat> mTaskThread;

    public:
        void loadImage(const std::string& pathToImg);
        void runWatershedSegmentation();
        void runCvWatershedSegmentation();
        void update();

};
