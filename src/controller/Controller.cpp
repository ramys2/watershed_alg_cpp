#include "controller/Controller.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "SFML/Graphics/RenderWindow.hpp"
#include "SFML/Graphics/Sprite.hpp"
#include "imgui-SFML.h"
#include "nfd.hpp"
#include "service/image_loader.hpp"
#include "service/image_service.hpp"

Controller::Controller(const sf::Vector2u &sfWindowSize)
    : mNumberOfMarkers(2), mGausianBlurSize(3), mMorphologyKernelSize(2),
      mCvNumberOfMarkers(2), mCvGausianBlurSize(3), mCvMorphologyKernelSize(2),
      mServiceIsProcessing(false), mDuration(0.0), mWatershedMethod(""),
      mOutputfilePath(generateTimestampPath()), mWindowSize(sfWindowSize)
{
    std::filesystem::path dir(OUTPUT_DIR);

    try
    {
        if (!std::filesystem::exists(dir))
        {
            std::filesystem::create_directories(dir);
        }
    }
    catch (const std::filesystem::filesystem_error &e)
    {
        // Handle potential permission issues or disk errors
        std::cerr << "Error creating directory: " << e.what() << std::endl;
    }
}

void Controller::loadImage()
{
    NFD::Guard nfdGuard;
    NFD::UniquePath outPath;
    nfdfilteritem_t filterItem[2] = {{"Image Files", "jpg,png,jpeg"},
                                     {"All Files", "*"}};

    nfdresult_t result = NFD::OpenDialog(outPath, filterItem, 2);

    if (result == NFD_OKAY)
    {
        if (mOriginalImgWin.isOpen())
        {
            mOriginalImgWin.close();
            mAppData.resetOriginalImage();
        }

        if (mSegmentedImgWin.isOpen())
        {
            mSegmentedImgWin.close();
            mAppData.resetSegmentedImage();
        }

        // outPath.get() gives you the const char*
        std::string pathToImg(outPath.get());
        // You can now use pathToImg for your logic
        cv::Mat image = image_loader::loadImage(pathToImg);

        if (image.empty())
        {
            return;
        }

        mAppData.updateOriginalImage(image);
    }
    else if (result == NFD_CANCEL)
    {
        return;
    }
    else
    {
        return;
    }
}

void Controller::runWatershedSegmentation()
{
    if (!mServiceIsProcessing)
    {
        cv::Mat clonedMatrix = mAppData.getOriginalMatrix().clone();

        if (clonedMatrix.empty())
        {
            return;
        }

        mServiceIsProcessing = true;
        mWatershedMethod = "custom_watershed";
        mStartTime = std::chrono::high_resolution_clock::now();
        mTaskFuture = std::async(std::launch::async,
                                 &image_service::watershedSegmentation,
                                 clonedMatrix, mNumberOfMarkers,
                                 mGausianBlurSize, mMorphologyKernelSize);
    }
}

void Controller::runCvWatershedSegmentation()
{
    if (!mServiceIsProcessing)
    {
        cv::Mat clonedMatrix = mAppData.getOriginalMatrix().clone();

        if (clonedMatrix.empty())
        {
            return;
        }

        mServiceIsProcessing = true;
        mWatershedMethod = "opencv_watershed";
        mStartTime = std::chrono::high_resolution_clock::now();
        mTaskFuture = std::async(std::launch::async,
                                 &image_service::cvWatershedSegmentation,
                                 clonedMatrix, mCvNumberOfMarkers,
                                 mCvGausianBlurSize, mCvMorphologyKernelSize);
    }
}

void Controller::update()
{
    if (mServiceIsProcessing && mTaskFuture.wait_for(std::chrono::seconds(0)) ==
                                    std::future_status::ready)
    {
        // Capture time
        mEndTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = mEndTime - mStartTime;
        mDuration = elapsed.count();

        // Update image model
        cv::Mat result = mTaskFuture.get();
        mAppData.updateSegmentedImage(result);
        writeTime();

        // Reset state
        mServiceIsProcessing = false;
        mStartTime = std::chrono::high_resolution_clock::time_point();
        mEndTime = std::chrono::high_resolution_clock::time_point();
        mDuration = 0.0;
        mWatershedMethod = "";
    }
}

void Controller::renderGuiElements()
{
    ImGui::SetNextWindowPos(CONTROL_PANEL_POSITION, ImGuiCond_Always);
    ImGui::SetNextWindowSize(
        ImVec2(CONTROL_PANEL_W, static_cast<float>(mWindowSize.y)),
        ImGuiCond_Always);

    ImGui::Begin("Control panel", nullptr, WINDOW_FLAGS);
    if (ImGui::Button("Load image"))
    {
        loadImage();
    }

    if (ImGui::Button("Run custom watershed"))
    {
        runWatershedSegmentation();
    }
    ImGui::SliderInt("Markers", &mNumberOfMarkers, 2, 253);
    if (ImGui::SliderInt("GBKS", &mGausianBlurSize, 3, 31))
    {
        if (mGausianBlurSize % 2 == 0)
        {
            mGausianBlurSize++;
        }

        if (mGausianBlurSize > 31)
        {
            mGausianBlurSize = 31;
        }
    }
    ImGui::SliderInt("MKS", &mMorphologyKernelSize, 2, 20);

    if (ImGui::Button("Run opencv watershed"))
    {
        runCvWatershedSegmentation();
    }

    ImGui::SliderInt("OpenCV Markers", &mCvNumberOfMarkers, 2, 253);
    if (ImGui::SliderInt("OpenCV GBKS", &mCvGausianBlurSize, 3, 31))
    {
        if (mCvGausianBlurSize % 2 == 0)
        {
            mCvGausianBlurSize++;
        }

        if (mCvGausianBlurSize > 31)
        {
            mCvGausianBlurSize = 31;
        }
    }
    ImGui::SliderInt("OpenCV MKS", &mCvMorphologyKernelSize, 2, 20);

    ImGui::End();
}

void Controller::renderImgWindows()
{
    renderImgWindow(mOriginalImgWin, mAppData.getOriginalTexture());
    renderImgWindow(mSegmentedImgWin, mAppData.getSegmentedTexture());
}

void Controller::renderImgWindow(sf::RenderWindow& window, const sf::Texture& texture)
{
    bool hasTexture = (texture.getSize().x != 0);

    if (hasTexture && !window.isOpen())
    {
        window.create(sf::VideoMode(1280, 720),
                               "High-Res Original View");
    }

    // 2. RENDERING & EVENTS: Only run if the window is currently open
    if (window.isOpen())
    {
        if (hasTexture)
        {
            window.clear(sf::Color::Black);

            sf::Sprite s(texture);

            // Scaled to fit for high-res handling
            float scale =
                std::min(static_cast<float>(window.getSize().x) /
                             texture.getSize().x,
                         static_cast<float>(window.getSize().y) /
                             texture.getSize().y);
            s.setScale(scale, scale);

            window.draw(s);
            window.display();
        }
    }
}

void Controller::processWinEvents()
{
    processWinEvent(mOriginalImgWin, mAppData.getOriginalTexture());
    processWinEvent(mSegmentedImgWin, mAppData.getSegmentedTexture());
}

void Controller::processWinEvent(sf::RenderWindow& window, const sf::Texture& texture)
{
    sf::Event event;
    while (window.pollEvent(event))
    {
        if (event.type == sf::Event::Closed)
        {
            window.close();
            if (&texture == &mAppData.getOriginalTexture())
            {
                mAppData.resetOriginalImage();
            }
            else
            {
                mAppData.resetSegmentedImage();
            }
        }
    }
}

std::string Controller::generateTimestampPath()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time), "%Y%m%d%H%M%S");
    return OUTPUT_DIR + "segmentation_runtime_" + ss.str() + ".csv";
}

void Controller::writeTime()
{
    std::ofstream file(mOutputfilePath, std::ios::app);
    int rows = mAppData.getSegmentedMatrix().rows;
    int cols = mAppData.getSegmentedMatrix().cols;
    if (file.is_open())
    {
        file << mWatershedMethod << ", " << cols << "x" << rows << ", "
             << mDuration << "\n";
        file.close();
    }
}
