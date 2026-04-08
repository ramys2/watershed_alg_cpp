#include "service/ImageService.hpp"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <queue>

cv::Mat ImageService::watershedSegmentation(const cv::Mat matrix,
                                            const int max_markers,
                                            const int gaussianBlurMatrixSize,
                                            const int kernelMatrixSize)
{
    std::cout << "Pre-processing image" << std::endl;
    cv::Mat preparedImg =
        preProcessForCustomWS(matrix, gaussianBlurMatrixSize, kernelMatrixSize);
    cv::Mat grayscaleImg = convertToGreyScale(matrix);
    cv::Mat markers = cv::Mat::zeros(matrix.size(), CV_8UC1);

    std::vector<std::pair<int, int>> local_mins =
        find_local_mins(preparedImg, max_markers); // TODO: Parametrize
    std::priority_queue<Pixel, std::vector<Pixel>, std::greater<Pixel>> pq;
    for (int i = 0; i < local_mins.size(); ++i)
    {
        int r = local_mins[i].first;
        int c = local_mins[i].second;
        uchar id = (uchar)(i) + FIRST_MARKER_ID;

        markers.at<uchar>(r, c) = id;
        pq.push({r, c, preparedImg.at<uchar>(r, c)});
    }

    while (!pq.empty())
    {
        Pixel current = pq.top();
        pq.pop();

        uchar current_label = markers.at<uchar>(current.r, current.c);

        for (int dr = -1; dr <= 1; ++dr)
        {
            for (int dc = -1; dc <= 1; ++dc)
            {
                if (dr == 0 && dc == 0)
                    continue;

                int nr = current.r + dr;
                int nc = current.c + dc;

                if (nr >= 0 && nr < markers.rows && nc >= 0 &&
                    nc < markers.cols)
                {
                    uchar &neighbor_label = markers.at<uchar>(nr, nc);

                    if (neighbor_label == 0)
                    {
                        neighbor_label = current_label;
                        pq.push({nr, nc, preparedImg.at<uchar>(nr, nc)});
                    }
                    else if (neighbor_label != current_label &&
                             neighbor_label != 1)
                    {
                        markers.at<uchar>(current.r, current.c) = 1;
                    }
                }
            }
        }
    }

    return postProcessingForCustomWS(grayscaleImg, markers);
}

cv::Mat ImageService::cvWatershedSegmentation(const cv::Mat matrix)
{
    if (matrix.empty())
        return matrix;

    // 1. Watershed requires 3-channel BGR for the internal math
    cv::Mat src;
    if (matrix.channels() == 4)
    {
        cv::cvtColor(matrix, src, cv::COLOR_RGBA2BGR);
    }
    else
    {
        src = matrix.clone();
    }

    // 2. Grayscale & Thresholding
    cv::Mat gray, thresh;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, thresh, 0, 255,
                  cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    // 3. Find Background (Dilate) and Foreground (Distance Transform)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat sure_bg;
    cv::dilate(thresh, sure_bg, kernel, cv::Point(-1, -1), 3);

    cv::Mat dist, sure_fg;
    cv::distanceTransform(thresh, dist, cv::DIST_L2, 5);

    double maxVal;
    cv::minMaxLoc(dist, nullptr, &maxVal);
    cv::threshold(dist, sure_fg, 0.9 * maxVal, 255, cv::THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8U);

    // 4. Markers & Unknown region
    cv::Mat markers, unknown;
    cv::connectedComponents(sure_fg, markers);
    cv::subtract(sure_bg, sure_fg, unknown);

    markers += 1;
    markers.setTo(0, unknown == 255);

    // 5. Execute Watershed
    cv::watershed(src, markers);

    // 6. Build the Grayscale-to-RGBA result
    cv::Mat result;
    // Start by making the whole image grayscale but in 4-channel format
    cv::cvtColor(gray, result, cv::COLOR_GRAY2RGBA);

    // 7. Draw the Blue lines
    result.setTo(cv::Scalar(0, 0, 255, 255), markers == -1);

    return result;
}

cv::Mat ImageService::preProcessForCustomWS(const cv::Mat &matrix,
                                            const int gaussianBlurMatrixSize,
                                            const int kernelMatrixSize)
{
    cv::Mat grayscale = convertToGreyScale(matrix);
    cv::GaussianBlur(grayscale, grayscale,
                     cv::Size(gaussianBlurMatrixSize, gaussianBlurMatrixSize),
                     0);

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelMatrixSize, kernelMatrixSize));
    cv::morphologyEx(grayscale, grayscale, cv::MORPH_GRADIENT, kernel);

    return grayscale;
}

cv::Mat ImageService::convertToGreyScale(const cv::Mat &matrix)
{
    cv::Mat greyImg;
    cv::cvtColor(matrix, greyImg, cv::COLOR_RGBA2GRAY);
    return greyImg;
}

cv::Mat ImageService::postProcessingForCustomWS(const cv::Mat &matrix,
                                                const cv::Mat &markers)
{
    cv::Mat rgbaImg;
    cv::cvtColor(matrix, rgbaImg, cv::COLOR_GRAY2RGBA);
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            if (markers.at<uchar>(i, j) == WATERSHED_LINE_MARKER)
            {
                rgbaImg.at<cv::Vec4b>(i, j) = cv::Vec4b(0, 0, 255, 255);
            }
        }
    }
    return rgbaImg;
}

std::vector<std::pair<int, int>>
ImageService::find_local_mins(const cv::Mat &img_mat, int max_markers)
{
    std::vector<Pixel> candidates;
    for (int r = 1; r < img_mat.rows - 1; ++r)
    {
        for (int c = 1; c < img_mat.cols - 1; ++c)
        {
            uchar center = img_mat.at<uchar>(r, c);

            if (center < img_mat.at<uchar>(r - 1, c - 1) &&
                center < img_mat.at<uchar>(r - 1, c) &&
                center < img_mat.at<uchar>(r - 1, c + 1) &&
                center < img_mat.at<uchar>(r, c - 1) &&
                center < img_mat.at<uchar>(r, c + 1) &&
                center < img_mat.at<uchar>(r + 1, c - 1) &&
                center < img_mat.at<uchar>(r + 1, c) &&
                center < img_mat.at<uchar>(r + 1, c + 1))
            {
                candidates.push_back({r, c, center});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end());

    std::vector<std::pair<int, int>> result;
    int count = std::min((int)candidates.size(), max_markers);
    for (int i = 0; i < count; ++i)
    {
        result.push_back({candidates[i].r, candidates[i].c});
    }

    return result;
}
