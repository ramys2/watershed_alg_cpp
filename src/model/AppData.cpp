#include "model/AppData.hpp"

void AppData::updateOriginalImage(const cv::Mat &matrix)
{
    mOriginalMatrix = matrix.clone();

    if (mOriginalTexture.getSize().x != mOriginalMatrix.cols || mOriginalTexture.getSize().y != mOriginalMatrix.rows)
    {
        mOriginalTexture.create(mOriginalMatrix.cols, mOriginalMatrix.rows);
    }

    mOriginalTexture.update(mOriginalMatrix.data);
}

void AppData::updateSegmentedImage(const cv::Mat &matrix)
{
    mSegmentedMatrix = matrix.clone();

    if (mSegmentedTexture.getSize().x != mSegmentedMatrix.cols || mSegmentedTexture.getSize().y != mSegmentedMatrix.rows)
    {
        mSegmentedTexture.create(mSegmentedMatrix.cols, mSegmentedMatrix.rows);
    }

    mSegmentedTexture.update(mSegmentedMatrix.data);
}
