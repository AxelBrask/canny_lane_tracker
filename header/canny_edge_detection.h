#include "types.h"


struct CannyEdgeDetection {

    double high_threshold_ = 150.0;
    double low_threshold_ = 100.0;
    virtual ~CannyEdgeDetection() = default;
    virtual Frame run(const Frame& frame) = 0;
    virtual void gaussianSmoothing(const Frame& frame, Frame& blur, double sigma) = 0;
    virtual std::vector<float> gaussianKernel1D(double sigma, int kernel_size) = 0;
    virtual void sobelFilter(const Frame& blur, Frame& mag, Frame& dir) = 0;
    /**
     * Perform Non-Maximum Suppression on the magnitude and direction frames. 
     * in order to thin the edges.
     */
    virtual void nonMaximumSuppression(const Frame& magnitude, const Frame& direction, Frame& nms) = 0;
    virtual void ensureBuffers(int w, int h) = 0;
    virtual int computeKernelSize(double sigma) = 0;
    virtual void ensureGaussianKernel(double sigma) = 0;
};
std::unique_ptr<CannyEdgeDetection> createCannyEdgeDetection();