#include "controller/Controller.hpp"

#include <future>
#include <functional>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "service/ImageService.hpp"

void Controller::loadImage() {
    // TODO: implement NDF library, replace char* by ndfchar_t*
    // and add checks for invalid path or file
    char* outPath;
    std::string pathToImg = outPath;

    if (pathToImg.empty()) {
        return;
    }

    cv::Mat image = cv::imread(pathToImg);

    mAppData.updateOriginalImage(image);
}

void Controller::runWatershedSegmentation() {
    if (!mAppData.serviceIsProcessing()) {
        cv::Mat clonedMatrix = mAppData.getOriginalMatrix().clone();

        mTaskThread = std::async(std::launch::async, &ImageService::watershedSegmentation, &mImageService, std::cref(clonedMatrix));
        mAppData.setServiceIsProcessing(true);
    }
}

void Controller::runCvWatershedSegmentation() {
    if (!mAppData.serviceIsProcessing()) {
        cv::Mat clonedMatrix = mAppData.getOriginalMatrix().clone();

        mTaskThread = std::async(std::launch::async, &ImageService::cvWatershedSegmentation, &mImageService, std::cref(clonedMatrix));
        mAppData.setServiceIsProcessing(true);
    }
}

void Controller::update() {
    if (mAppData.serviceIsProcessing() && mTaskThread.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        cv::Mat result = mTaskThread.get();
        mAppData.updateSegmentedImage(result);
        mAppData.setServiceIsProcessing(false);
    }
}
