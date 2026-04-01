#include "controller/Controller.hpp"

#include <future>
#include <functional>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "nfd.hpp"
#include "service/ImageService.hpp"

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
        cv::Mat greyImg = mImageService.convertToGreyScale(clonedMatrix);

        mTaskThread = std::async(std::launch::async, &ImageService::watershedSegmentation, &mImageService, std::cref(clonedMatrix));
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
        cv::Mat greyImg = mImageService.convertToGreyScale(clonedMatrix);

        mTaskThread = std::async(std::launch::async, &ImageService::cvWatershedSegmentation, &mImageService, std::cref(greyImg));
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

const sf::Texture &Controller::retreiveOriginalImage() const
{
    const sf::Texture &img = mAppData.getOriginalTexture();
    if (img.getSize().x == 0 || img.getSize().y == 0)
    {
        static const sf::Texture emptyTexture;
        return emptyTexture;
    }

    return img;
}
