#include "canny_edge_detection.h"

struct canny_edge_detection_impl : public CannyEdgeDetection {

    CannyEdgeConfig config_;
    // static sobel matrices
    static const Matrix sobel_x;
    static const Matrix sobel_y;
    Frame tmp{0,0};
    Frame blur{0,0};
    Frame mag{0,0};
    Frame dir{0,0};
    Frame nms{0,0};
    std::vector<float> gaussian_kernel;
    int gaussian_kernel_size_ = 0;

    // Predifine mask for lower imgage
    int mask_height = 0;

    void ensureBuffers(int w, int h) {
        if (tmp.width != w || tmp.height != h) {
            tmp = Frame(w, h);
            blur = Frame(w, h);
            mag = Frame(w, h);
            dir = Frame(w, h);
            nms = Frame(w, h);
        }
    }

    Frame run(const Frame& frame) override {
        // Implement Canny edge detection algorithm
        ensureBuffers(frame.width, frame.height);
        mask_height = frame.height;
        gaussianSmoothing(frame, blur);
        sobelFilter(blur, mag, dir);
        nonMaximumSuppression(mag, dir, nms);
        return nms;
    }

    void gaussianSmoothing(const Frame& frame, Frame& blur) {
        // Implement Gaussian smoothing
        ensureGaussianKernel();
        int y0 = ImageMask::getMaskStartY(frame.height);


        //Horizontal pass
        int half_size = gaussian_kernel_size_ / 2;
        int y_blur_start = std::max(0,y0);
        int y_tmp_start = std::max(0,y_blur_start-half_size);

        for (int y = y_tmp_start; y < frame.height; ++y) {
            for (int x = 0; x < frame.width; ++x) {
                float sum = 0.0f;

                int x_start  = std::max(0, x - half_size);
                int x_end    = std::min(frame.width - 1, x + half_size);

                for (int xx = x_start; xx <= x_end; ++xx) {
                    int k = xx - x + half_size;
                    sum += frame.at(xx, y) * gaussian_kernel[k];
                }
                tmp.at(x, y) = static_cast<uint8_t>(sum);
            }
        }

        // Vertical pass
        half_size = gaussian_kernel_size_ / 2;
        for (int y = y_blur_start; y < frame.height; ++y) {
            for (int x = 0; x < frame.width; ++x) {
                float sum = 0.0f;
                
                for (int k = -half_size; k <= half_size; ++k) {
                    int yy = std::clamp(y + k, 0, frame.height - 1);
                    sum += tmp.at(x, yy) * gaussian_kernel[k + half_size];
                }
                blur.at(x, y) = static_cast<uint8_t>(sum);
            }
        }

    }
    int computeKernelSize(double sigma) {
        int kernel_size = static_cast<int>(std::ceil(3.0 * sigma));
        if (kernel_size % 2 == 0) kernel_size++; // Ensure odd size
        kernel_size= std::clamp(kernel_size, 1, 15);
        return kernel_size;
    }

    void ensureGaussianKernel() {
        int kernel_size = computeKernelSize(config_.sigma);
        if (kernel_size != gaussian_kernel_size_) {
            gaussian_kernel = gaussianKernel1D(config_.sigma, kernel_size);
            gaussian_kernel_size_ = kernel_size;
        }
    }

    std::vector<float> gaussianKernel1D(double sigma, int kernel_size) {
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

    void sobelFilter(const Frame& frame, Frame& mag, Frame& dir) {
        int y0 = ImageMask::getMaskStartY(frame.height);
        for (int y = y0; y < frame.height - 1; ++y) {
            for (int x = 1; x < frame.width - 1; ++x) {
                float gx = 0.0f;
                float gy = 0.0f;

                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        gx += frame.at(x + kx, y + ky) * sobel_x[ky + 1][kx + 1];
                        gy += frame.at(x + kx, y + ky) * sobel_y[ky + 1][kx + 1];
                    }
                }

                float magnitude = std::sqrt(gx * gx + gy * gy);
                magnitude = std::clamp(magnitude, 0.0f, 255.0f);
                //compute approximation of
                float angle = std::atan2(gy, gx);

                float angle_deg = angle * 180.0f / CV_PI;
                if (angle_deg < 0) {
                    angle_deg += 180.0f;
                }
                dir.at(x, y) = static_cast<uint8_t>((angle_deg / 180.0f) * 255.0f);


                mag.at(x, y) = static_cast<uint8_t>(magnitude);
            }
        }
    }

    void nonMaximumSuppression(const Frame& magnitude, const Frame& direction, Frame& nms) {
        int y0 = magnitude.height / 2;
        for (int y = y0; y < magnitude.height - 1; ++y) {
            for (int x = 1; x < magnitude.width - 1; ++x) {
                float angle = (direction.at(x, y)) / 255.0f * 180.0f;
                float mag = magnitude.at(x, y);
                float q = 255.0f;

                uint8_t neighbor1 = 0, neighbor2 = 0;
                int direction_index = static_cast<int>((angle + 22.5) / 45.0) % 4;

                switch(direction_index) {
                    case 0: // 0 degrees
                        neighbor1 = magnitude.at(x + 1, y);
                        neighbor2 = magnitude.at(x - 1, y);
                        break;
                    case 1: // 45 degrees
                        neighbor1 = magnitude.at(x + 1, y - 1);
                        neighbor2 = magnitude.at(x - 1, y + 1);
                        break;
                    case 2: // 90 degrees
                        neighbor1 = magnitude.at(x, y - 1);
                        neighbor2 = magnitude.at(x, y + 1);
                        break;
                    case 3: // 135 degrees
                        neighbor1 = magnitude.at(x - 1, y - 1);
                        neighbor2 = magnitude.at(x + 1, y + 1);
                        break;
                }
                if (mag >= neighbor1 && mag >= neighbor2) {
                    nms.at(x, y) = static_cast<uint8_t>(mag);
                } else {
                    nms.at(x, y) = 0;
                }
            // Hysterias Thresholding
            if (mag >= config_.high_threshold) {
                nms.at(x, y) = static_cast<uint8_t>(mag);
            } else if (mag < config_.low_threshold) {
                nms.at(x, y) = 0;
            }
        }
        
        }
    }
};

const Matrix canny_edge_detection_impl::sobel_x = {
    { -1, 0, 1 },
    { -2, 0, 2 },
    { -1, 0, 1 }
};

const Matrix canny_edge_detection_impl::sobel_y = {
    {  1,  2,  1 },
    {  0,  0,  0 },
    { -1, -2, -1 }
};

std::unique_ptr<CannyEdgeDetection> createCannyEdgeDetection() {
    return std::make_unique<canny_edge_detection_impl>();
}