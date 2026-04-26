#pragma once

#include <fstream>
#include <future>
#include <SFML/Config.hpp>
#include <string>
#include <chrono>

#include "SFML/Graphics/RenderWindow.hpp"
#include "SFML/Graphics/Texture.hpp"
#include "SFML/System/Vector2.hpp"
#include "service/image_service.hpp"
#include "model/ImageData.hpp"
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

    std::ofstream mFile;

    sf::Vector2u mWindowSize;

    float mOrgImgW;
    int mNumberOfMarkers;
    int mGausianBlurSize;
    int mMorphologyKernelSize;

    int mCvNumberOfMarkers;
    int mCvGausianBlurSize;
    int mCvMorphologyKernelSize;

    // ================================================
    // Image data model
    //  ===============================================
    ImageData mAppData;
    sf::RenderWindow mOriginalImgWin;
    sf::RenderWindow mSegmentedImgWin;

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
    Controller(const sf::Vector2u &sfWindowSize);
    ~Controller();
    // Polls for result when segmentation is running
    void update();
    // Renders window with GUI Elements
    void renderGuiElements();

    // Renders windows with loaded and segmented image if they are avialable
    void renderImgWindows();
    void processWinEvents();

    const sf::Vector2u& getWindowSize() const { return mWindowSize; }
    void setWindowSize(const sf::Vector2u &sfWindowSize) { mWindowSize = sfWindowSize; }

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
    void renderImgWindow(sf::RenderWindow& window, const sf::Texture& texture);
    void processWinEvent(sf::RenderWindow& window, const sf::Texture& texture);
};
