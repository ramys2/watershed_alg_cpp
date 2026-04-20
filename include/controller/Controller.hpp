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
    // ================================================
    // UI components variables
    //  ===============================================
    static constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoMove |
                                                    ImGuiWindowFlags_NoResize |
                                                    ImGuiWindowFlags_NoCollapse |
                                                    ImGuiWindowFlags_NoBringToFrontOnFocus;
    static constexpr float CONTROL_PANEL_W = 400.0f;
    static constexpr ImVec2 CONTROL_PANEL_POSITION = ImVec2(0, 0);
    static constexpr ImVec2 ORIGINAL_IMG_POSITION = ImVec2(CONTROL_PANEL_W, 0);
    static constexpr std::string OUTPUT_DIR = "statistics/";

    const std::string mOutputfilePath;

    float mOrgImgW;

    int mNumberOfMarkers;
    int mGausianBlurSize;
    int mMorphologyKernelSize;

    int mCvNumberOfMarkers;
    int mCvGausianBlurSize;
    int mCvMorphologyKernelSize;

    ImageService mImageService;
    ImageLoader mImageLoader;

    // ================================================
    // Image data model
    //  ===============================================
    AppData mAppData;

    // ================================================
    // Processing state variables
    //  ===============================================
    std::future<cv::Mat> mTaskFuture;
    double mDuration;
    bool mServiceIsProcessing;
    std::string mWatershedMethod;
    std::chrono::time_point<std::chrono::high_resolution_clock> mStartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> mEndTime;

public:
    Controller();
    ~Controller() = default;
    // Polls for result when segmentation is running
    void update();
    // Renders window with GUI Elements
    void renderGuiElements(const sf::Vector2u &sfWindowSize);
    // Renders loaded image if it is available
    void renderOriginalImage(const sf::Vector2u& sfWindowSize);
    // Render segmented image if it is avialable
    void renderSegmentedlImage(const sf::Vector2u& sfWindowSize);

private:
    // Generates output file with timestamp
    static std::string generateTimestampPath();
    // Invokes service to load image
    void loadImage();
    // Invokes service to run manual implementation of watershed
    void runWatershedSegmentation();
    // Invokes service to run opencv implementation of watershed
    void runCvWatershedSegmentation();
    void writeTime();
};
