#pragma once

#include <future>
#include <SFML/Config.hpp>
#include <string>
#include <chrono>

#include "service/ImageLoader.hpp"
#include "service/ImageService.hpp"
#include "model/AppData.hpp"
#include "imgui.h"

class Controller final
{
private:
    static constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoMove |
                                                    ImGuiWindowFlags_NoResize |
                                                    ImGuiWindowFlags_NoCollapse |
                                                    ImGuiWindowFlags_NoBringToFrontOnFocus;
    static constexpr float CONTROL_PANEL_W = 400.0f;
    static constexpr ImVec2 CONTROL_PANEL_POSITION = ImVec2(0, 0);
    static constexpr ImVec2 ORIGINAL_IMG_POSITION = ImVec2(CONTROL_PANEL_W, 0);

    float mOrgImgW = 0;

    int mNumberOfMarkers = 2;
    int mGausianBlurSize = 3;
    int mMorphologyKernelSize = 2;

    int mCvNumberOfMarkers = 2;
    int mCvGausianBlurSize = 3;
    int mCvMorphologyKernelSize = 2;

    ImageService mImageService;
    ImageLoader mImageLoader;

    AppData mAppData;

    std::future<cv::Mat> mTaskFuture;
    bool mServiceIsProcessing = false;
    std::string mWatershedMethod;
    std::chrono::time_point<std::chrono::high_resolution_clock> mStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> mEndTime;
    double mDuration = 0.0;

public:
    // Polls for result when segmentation is running
    void update();
    void renderGuiElements(const sf::Vector2u &sfWindowSize);
    void renderOriginalImage(const sf::Vector2u& sfWindowSize);
    void renderSegmentedlImage(const sf::Vector2u& sfWindowSize);

private:
    // Invokes service to load image
    void loadImage();
    // Invokes service to run manual implementation of watershed
    void runWatershedSegmentation();
    // Invokes service to run opencv implementation of watershed
    void runCvWatershedSegmentation();
};
