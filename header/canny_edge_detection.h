#include "types.h"


struct CannyEdgeDetection {
    virtual ~CannyEdgeDetection() = default;
    virtual Frame run(const Frame& frame) = 0;
    virtual Frame gaussianSmoothing(const Frame& frame, double sigma) = 0;
    virtual std::vector<float> gaussianKernel1D(double sigma, int kernel_size) = 0;
};
std::unique_ptr<CannyEdgeDetection> createCannyEdgeDetection();