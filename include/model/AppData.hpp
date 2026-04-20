#pragma once

#include <opencv2/core.hpp>
#include <SFML/Graphics.hpp>

class AppData final
{
private:
    cv::Mat mOriginalMatrix;
    cv::Mat mSegmentedMatrix;
    sf::Texture mOriginalTexture;
    sf::Texture mSegmentedTexture;

public:
    const cv::Mat &getOriginalMatrix() const { return mOriginalMatrix; }
    const cv::Mat &getSegmentedMatrix() const { return mSegmentedMatrix; }
    const sf::Texture &getOriginalTexture() const { return mOriginalTexture; }
    const sf::Texture &getSegmentedTexture() const { return mSegmentedTexture; }
    // Updates Matrix and corresponding sf::Texture
    void updateOriginalImage(const cv::Mat &matrix);
    // Updates Matrix and corresponding sf::Texture
    void updateSegmentedImage(const cv::Mat &matrix);
};
