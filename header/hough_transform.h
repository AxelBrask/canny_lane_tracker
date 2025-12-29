#include "types.h"

struct HoughTransformConfig {
    int lineThreshold = 50;
    double minTheta = -90.0;
    double maxTheta = 90.0;
    double angleStep = 1.0;
    double rhoStep = 1.0;
    int numberOfLines = 5;
};

struct HoughTransform {
    virtual ~HoughTransform() = default;
    virtual Frame run(const Frame& edges) = 0;

    virtual const std::vector<HoughLine>& getDetectedLines() const = 0;
};

std::unique_ptr<HoughTransform> createHoughTransform();