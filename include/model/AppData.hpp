#pragma once

#include <opencv2/core.hpp>
#include <SFML/Graphics.hpp>

class AppData final {
    private:
        cv::Mat mOriginalMatrix;
        cv::Mat mSegmentedMatrix;
        sf::Texture mOriginalTexture;
        sf::Texture mSegmentedTexture;
        bool mServiceIsProcessing;

    public:
        void updateOriginalImage(const cv::Mat matrix);
        void updateSegmentedImage(const cv::Mat matrix);
        bool serviceIsProcessing() const { return mServiceIsProcessing; }
        void setServiceIsProcessing(bool isProcessing) { mServiceIsProcessing = isProcessing; }
};
