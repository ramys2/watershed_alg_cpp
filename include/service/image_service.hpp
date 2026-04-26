#include <opencv2/core.hpp>

namespace image_service
{
    cv::Mat watershedSegmentation(const cv::Mat &matrix, const int max_markers, const int gaussianBlurMatrixSize, const int kernelMatrixSize);
    cv::Mat cvWatershedSegmentation(const cv::Mat &matrix, const int max_markers, const int gaussianBlurMatrixSize, const int kernelMatrixSize);
}
