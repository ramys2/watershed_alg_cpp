#include "controller/Controller.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "imgui-SFML.h"
#include "nfd.hpp"
#include "service/ImageService.hpp"
#include "service/image_loader.hpp"

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
        mTaskFuture =
            std::async(std::launch::async, &ImageService::watershedSegmentation,
                       &mImageService, clonedMatrix, mNumberOfMarkers,
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
        mTaskFuture = std::async(
            std::launch::async, &ImageService::cvWatershedSegmentation,
            &mImageService, clonedMatrix, mCvNumberOfMarkers,
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
        if (mGausianBlurSize % 2 == 0)
        {
            mGausianBlurSize++;
        }

        if (mGausianBlurSize > 31)
        {
            mGausianBlurSize = 31;
        }
    }
    ImGui::SliderInt("OpenCV MKS", &mCvMorphologyKernelSize, 2, 20);

    ImGui::End();
}

void Controller::renderOriginalImage()
{
    const sf::Texture &texture = mAppData.getOriginalTexture();
    if (texture.getSize().x != 0 && texture.getSize().y != 0)
    {
        mOrgImgW = (static_cast<float>(mWindowSize.x) - CONTROL_PANEL_W) / 2;
        ImGui::SetNextWindowPos(ORIGINAL_IMG_POSITION, ImGuiCond_Always);
        ImGui::SetNextWindowSize(
            ImVec2(mOrgImgW, static_cast<float>(mWindowSize.y)),
            ImGuiCond_Always);

        ImGui::Begin("Original Image", nullptr, WINDOW_FLAGS);
        ImGui::Image(texture);
        ImGui::End();
    }
}

void Controller::renderSegmentedlImage()
{
    const sf::Texture &texture = mAppData.getSegmentedTexture();
    if (texture.getSize().x != 0 && texture.getSize().y != 0)
    {
        float segImgW =
            static_cast<float>(mWindowSize.x) - CONTROL_PANEL_W - mOrgImgW;
        float segImgX = mOrgImgW + CONTROL_PANEL_W;

        ImGui::SetNextWindowPos(ImVec2(segImgX, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(
            ImVec2(segImgW, static_cast<float>(mWindowSize.y)),
            ImGuiCond_Always);

        ImGui::Begin("Segmented Image", nullptr, WINDOW_FLAGS);
        ImGui::Image(texture);
        ImGui::End();
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
