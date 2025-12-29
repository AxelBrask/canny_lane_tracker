#include "types.h"

struct HoughTransform {
    Matrix accumulator;
    int lineThreshold_ = 1;
    Frame lines{0,0};
    double minTheta = -90;
    double maxTheta = 90;
    double angleStep = 1.0;
    static const int number_of_lines_ = 5;
    std::vector<HoughLine> top;

    std::vector<double> rhos_;
    double rho_min_ = 0.0;
    double rho_step_ = 1.0;
    size_t rho_bins_ = 0;
    std::vector<double> thetas_rad_;
    std::vector<double> cos_t_;
    std::vector<double> sin_t_;
    double cachedMinTheta_ = 1e9;
    double cachedMaxTheta_ = 1e9;
    double cachedAngleStep_ = 1e9;

    virtual ~HoughTransform() = default;
    virtual Frame run(const Frame& edges) = 0;
    virtual void ensureAccumulatorSize(int width, int height) = 0;
    virtual void transformEdges(const Frame& edges, Frame& lines) = 0;
    virtual void precomputeSinCosRho(int width, int height) = 0;
    virtual inline void votePixel(int x, int y) = 0;
    virtual inline int rhoIndex(double rho) = 0;
    virtual void drawLines(Frame& lines, double rho, double theta) = 0;
    virtual void drawLine(Frame& frame, int x0, int y0, int x1, int y1) = 0;
    virtual bool isLocalMaximum(size_t r_idx, size_t t_idx) = 0;
    virtual bool checkLineValidity(size_t r_idx, size_t t_idx) = 0;

    static void insertTopN(std::vector<HoughLine>& top, float votes, double rho, double theta) {
        if (votes <= top.back().votes) return;
        top.back() = HoughLine{votes, rho, theta};
        std::sort(top.begin(), top.end(), [](const HoughLine& a, const HoughLine& b) {
            return a.votes > b.votes;
        });
    }
};

std::unique_ptr<HoughTransform> createHoughTransform();