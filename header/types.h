#pragma once
#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>

struct Frame {
    int width;
    int height;
    std::vector<std::uint8_t> pixels; // Assuming grayscale for simplicity

    Frame(int w, int h) : width(w), height(h), pixels(w * h) {}

    uint8_t& at(int x, int y) {
        return pixels[y * width + x];
    }
    const uint8_t& at(int x, int y) const {
        return pixels[y * width + x];
    }

    static Frame fromMat(const cv::Mat& mat) {
        Frame frame(mat.cols, mat.rows);

        cv::Mat gray;
        if (mat.channels() == 3) {
            cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
        } else{
            gray = mat;
        }

        std::memcpy(frame.pixels.data(), gray.data, frame.width * frame.height);
        return frame;
    }

    static cv::Mat toMat(const Frame& frame) {
        return cv::Mat(frame.height, frame.width, CV_8UC1, const_cast<uint8_t*>(frame.pixels.data()));
    }
};

using Matrix = std::vector<std::vector<float>>;
struct Edges {};
struct HoughLine { 
    float votes=0;
    double rho=0;
    double theta=0;
};

struct TrackedState {};

class ImageMask {
public:
    static bool isInLowerRegion(int x, int y, int width, int height) {
        return y >= height / 2 - 1;
    }

    static int getMaskStartY(int height) {
        return height / 2 - 1;
    }
};