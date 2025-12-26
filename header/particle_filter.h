#include "types.h"

struct HoughTransform {
    virtual ~HoughTransform() = default;
    virtual Lines run(const Edges& edges) = 0;
};