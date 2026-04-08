#include "controller/Controller.hpp"

#include <future>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "nfd.hpp"
#include "service/ImageService.hpp"
#include "imgui-SFML.h"

void Controller::loadImage()
{
    // TODO: implement NDF library, replace char* by ndfchar_t*
    // and add checks for invalid path or file
    NFD::Guard nfdGuard;
    NFD::UniquePath outPath;
    nfdfilteritem_t filterItem[2] = {{"Image Files", "jpg,png,jpeg"}, {"All Files", "*"}};

    nfdresult_t result = NFD::OpenDialog(outPath, filterItem, 2);

    if (result == NFD_OKAY)
    {
        // outPath.get() gives you the const char*
        std::string pathToImg(outPath.get());
        // You can now use pathToImg for your logic
        cv::Mat image = mImageLoader.loadImage(pathToImg);

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
    if (!mAppData.serviceIsProcessing())
    {
        cv::Mat clonedMatrix = mAppData.getOriginalMatrix().clone();

        if (clonedMatrix.empty())
        {
            return;
        }

        mTaskThread = std::async(std::launch::async, &ImageService::watershedSegmentation, &mImageService, clonedMatrix, mNumberOfMarkers, mGausianBlurSize, mMorphologyKernelSize);
        mAppData.setServiceIsProcessing(true);
    }
}

void Controller::runCvWatershedSegmentation()
{
    if (!mAppData.serviceIsProcessing())
    {
        cv::Mat clonedMatrix = mAppData.getOriginalMatrix().clone();

        if (clonedMatrix.empty())
        {
            return;
        }

        mTaskThread = std::async(std::launch::async, &ImageService::cvWatershedSegmentation, &mImageService, clonedMatrix);
        mAppData.setServiceIsProcessing(true);
    }
}

void Controller::update()
{
    if (mAppData.serviceIsProcessing() && mTaskThread.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
    {
        cv::Mat result = mTaskThread.get();
        mAppData.updateSegmentedImage(result);
        mAppData.setServiceIsProcessing(false);
    }
}

void Controller::renderGuiElements(const sf::Vector2u &sfWindowSize)
{
    ImGui::SetNextWindowPos(CONTROL_PANEL_POSITION, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(CONTROL_PANEL_W, static_cast<float>(sfWindowSize.y)), ImGuiCond_Always);

    ImGui::Begin("Control panel", nullptr, WINDOW_FLAGS);
    if (ImGui::Button("Load image"))
    {
        loadImage();
    }

    if (ImGui::Button("Run custom watershed"))
    {
        runWatershedSegmentation();
    }
    if (ImGui::Button("Run opencv watershed"))
    {
        runCvWatershedSegmentation();
    }
    ImGui::SliderInt("Markers", &mNumberOfMarkers, 2, 253);
    if (ImGui::SliderInt("Gaussian Kernel Blur Size", &mGausianBlurSize, 3, 31))
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
    ImGui::SliderInt("Morphology kernel size", &mMorphologyKernelSize, 2, 20);
    ImGui::End();
}

void Controller::renderOriginalImage(const sf::Vector2u &sfWindowSize)
{
    const sf::Texture &texture = mAppData.getOriginalTexture();
    if (texture.getSize().x != 0 && texture.getSize().y != 0)
    {
        mOrgImgW = (static_cast<float>(sfWindowSize.x) - CONTROL_PANEL_W) / 2;
        ImGui::SetNextWindowPos(ORIGINAL_IMG_POSITION, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(mOrgImgW, static_cast<float>(sfWindowSize.y)), ImGuiCond_Always);

        ImGui::Begin("Original Image", nullptr, WINDOW_FLAGS);
        ImGui::Image(texture);
        ImGui::End();
    }
}

void Controller::renderSegmentedlImage(const sf::Vector2u &sfWindowSize)
{
    const sf::Texture &texture = mAppData.getSegmentedTexture();
    if (texture.getSize().x != 0 && texture.getSize().y != 0)
    {
        float segImgW = static_cast<float>(sfWindowSize.x) - CONTROL_PANEL_W - mOrgImgW;
        float segImgX = mOrgImgW + CONTROL_PANEL_W;

        ImGui::SetNextWindowPos(ImVec2(segImgX, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(segImgW, static_cast<float>(sfWindowSize.y)), ImGuiCond_Always);

        ImGui::Begin("Segmented Image", nullptr, WINDOW_FLAGS);
        ImGui::Image(texture);
        ImGui::End();
    }
}
