#include "canny_edge_detection.h"

struct canny_edge_detection_impl : public CannyEdgeDetection {
    Frame run(const Frame& frame) override {
        // Implement Canny edge detection algorithm
        Frame blur_frame = gaussianSmoothing(frame, 1.5);
        return blur_frame;
    }

    Frame gaussianSmoothing(const Frame& frame, double sigma) override {
        // Implement Gaussian smoothing
        // Calculate appropriate kernel size: rule of thumb is 6*sigma + 1, make it odd
        int kernel_size = static_cast<int>(std::ceil(6.0 * sigma));
        if (kernel_size % 2 == 0) kernel_size++; // Ensure odd size
        kernel_size = std::max(3, kernel_size);   // Minimum size of 3
        std::vector<float> kernel = gaussianKernel1D(sigma, kernel_size);

        Frame temp_frame(frame.width, frame.height);
        Frame result_frame(frame.width, frame.height);

        //Horizontal pass
        int half_size = kernel_size / 2;
        for (int y = 0; y < frame.height; ++y) {
            for (int x = 0; x < frame.width; ++x) {
                float sum = 0.0f;
                for (int k = -half_size; k <= half_size; ++k) {
                    int xx = std::clamp(x + k, 0, frame.width - 1);
                    sum += frame.at(xx, y) * kernel[k + half_size];
                }
                temp_frame.at(x, y) = static_cast<uint8_t>(sum);
            }
        }

        // Vertical pass
        half_size = kernel_size / 2;
        for (int y = 0; y < frame.height; ++y) {
            for (int x = 0; x < frame.width; ++x) {
                float sum = 0.0f;
                for (int k = -half_size; k <= half_size; ++k) {
                    int yy = std::clamp(y + k, 0, frame.height - 1);
                    sum += temp_frame.at(x, yy) * kernel[k + half_size];
                }
                result_frame.at(x, y) = static_cast<uint8_t>(sum);
            }
        }

        return result_frame;
    }


    std::vector<float> gaussianKernel1D(double sigma, int kernel_size) override {
        std::vector<float> kernel(kernel_size);
        int half_size = kernel_size / 2;
        float sum = 0.0f;

        for (int i = -half_size; i <= half_size; ++i) {
            float value = std::exp(-(i * i) / (2 * sigma * sigma));
            kernel[i + half_size] = value;
            sum += value;
        }

        // Normalize the kernel
        for (auto& value : kernel) {
            value /= sum;
        }

        return kernel;
    }
};

std::unique_ptr<CannyEdgeDetection> createCannyEdgeDetection() {
    return std::make_unique<canny_edge_detection_impl>();
}