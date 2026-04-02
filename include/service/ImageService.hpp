#pragma once

#include <opencv2/core.hpp>

class ImageService final
{
public:
    cv::Mat watershedSegmentation(const cv::Mat matrix, const int max_markers, const int gaussianBlurMatrixSize, const int kernelMatrixSize);
    cv::Mat cvWatershedSegmentation(const cv::Mat matrix);

private:
    struct Pixel
    {
        int r, c;
        uchar value;

        bool operator<(const Pixel &other) const
        {
            return value < other.value;
        }

        bool operator>(const Pixel &other) const
        {
            return value > other.value;
        }
    };

    static constexpr uchar UNLABELED_MARKER = 0;
    static constexpr uchar WATERSHED_LINE_MARKER = 1;
    static constexpr uchar FIRST_MARKER_ID = 2;

    cv::Mat preProcessForCustomWS(const cv::Mat &matrix, const int gaussianBlurMatrix, const int kernelMatrixSize);
    cv::Mat convertToGreyScale(const cv::Mat &matrix);
    cv::Mat postProcessingForCustomWS(const cv::Mat &matrix, const cv::Mat &markers);
    std::vector<std::pair<int, int>> find_local_mins(const cv::Mat &img_mat, int max_markers);
};
