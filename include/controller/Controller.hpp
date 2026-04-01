#pragma once

#include <future>
#include <SFML/Config.hpp>

#include "service/ImageLoader.hpp"
#include "service/ImageService.hpp"
#include "model/AppData.hpp"
#include "imgui.h"
#include "imgui-SFML.h"

class Controller final
{
private:
    static constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoMove |
                                                    ImGuiWindowFlags_NoResize |
                                                    ImGuiWindowFlags_NoCollapse |
                                                    ImGuiWindowFlags_NoBringToFrontOnFocus;
    static constexpr float CONTROL_PANEL_W = 300.0f;
    static constexpr ImVec2 CONTROL_PANEL_POSITION = ImVec2(0, 0);
    static constexpr ImVec2 ORIGINAL_IMG_POSITION = ImVec2(CONTROL_PANEL_W, 0);

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

    void renderGuiElements(const sf::Vector2u &sfWindowSize);
    void renderOriginalImage(const sf::Vector2u& sfWindowSize);
};
