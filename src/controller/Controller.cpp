#include "controller/Controller.hpp"

#include <future>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "service/ImageService.hpp"

void Controller::loadImage(const std::string& pathToImg) {
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
