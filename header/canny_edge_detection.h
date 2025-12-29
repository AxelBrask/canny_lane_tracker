#pragma once
#include "types.h"
#include <memory>

struct CannyEdgeConfig {
    double high_threshold = 150.0;
    double low_threshold = 100.0;
    double sigma = 2.0;
};

struct CannyEdgeDetection {

    virtual ~CannyEdgeDetection() = default;
    virtual Frame run(const Frame& frame) = 0;
};

std::unique_ptr<CannyEdgeDetection> createCannyEdgeDetection();