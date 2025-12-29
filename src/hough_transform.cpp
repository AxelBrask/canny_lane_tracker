#include "hough_transform.h"

struct hough_transform_impl : public HoughTransform {
    HoughTransformConfig config_;
    Matrix accumulator;
    Frame lines{0,0};


    std::vector<double> rhos_;
    std::vector<double> thetas_rad_;
    std::vector<double> cos_t_;
    std::vector<double> sin_t_;
    double rho_min_ = 0.0;
    size_t rho_bins_ = 0;
    double cachedMinTheta_ = 1e9;
    double cachedMaxTheta_ = 1e9;
    double cachedAngleStep_ = 1e9;

    std::vector<HoughLine> top;

    void ensureBuffers(int w, int h) {
        if (lines.width != w || lines.height != h) {
            lines = Frame(w, h);
        }
    }

    void precomputeSinCosRho(int width, int height) {
        // early exit if already computed for these parameters
        if (cachedMinTheta_ == config_.minTheta && cachedMaxTheta_ == config_.maxTheta && cachedAngleStep_ == config_.angleStep) {
            return;
        }
        cachedMinTheta_ = config_.minTheta;
        cachedMaxTheta_ = config_.maxTheta;
        cachedAngleStep_ = config_.angleStep;

        thetas_rad_.clear();
        cos_t_.clear();
        sin_t_.clear();
        rhos_.clear();

        for (double theta = config_.minTheta; theta <= config_.maxTheta; theta += config_.angleStep) {
            double rad = theta * CV_PI / 180.0;
            thetas_rad_.push_back(rad);
            cos_t_.push_back(std::cos(rad));
            sin_t_.push_back(std::sin(rad));
        }
        const double diagLen = std::sqrt(width * width + height * height);
        for (double r = -diagLen; r <= diagLen; r += config_.rhoStep) {
            rhos_.push_back(r);
        }
        rho_bins_ = rhos_.size();
        rho_min_ = -diagLen;
    }

    void ensureAccumulatorSize(int width, int height) {

        precomputeSinCosRho(width, height);
        const size_t R = rho_bins_;
        const size_t T = thetas_rad_.size();

        if (accumulator.size() != R || (R > 0 && accumulator[0].size() != T)) {
            accumulator.assign(R, std::vector<float>(T, 0.0f));
        } else {
            for (auto& row : accumulator) std::fill(row.begin(), row.end(), 0.0f);
        }
        
    }

    Frame run(const Frame& edges) override {
        ensureBuffers(edges.width, edges.height);
        std::fill(lines.pixels.begin(), lines.pixels.end(), 0);
        ensureAccumulatorSize(edges.width, edges.height);
        transformEdges(edges, lines);
        return lines;
    }

    inline int rhoIndex(double rho) {
        return (int)std::lround((rho - rho_min_) / config_.rhoStep);
    }

    inline void votePixel(int x, int y) {
        for (size_t theta_idx = 0; theta_idx < thetas_rad_.size(); ++theta_idx) {
            double rho = x * cos_t_[theta_idx] + y * sin_t_[theta_idx];
            // find the closest rho index
            int rho_idx = rhoIndex(rho);
            if (rho_idx < accumulator.size()) {
                accumulator[(size_t)rho_idx][theta_idx] += 1.0f;
            }
        }
    }

    void transformEdges(const Frame& edges, Frame& lines) {
        top.clear();
        top.resize(config_.numberOfLines, HoughLine{0,0,0});

        int y0 = ImageMask::getMaskStartY(edges.height);
        for (int y = y0; y < edges.height; ++y) {
            for (int x = 0; x < edges.width; ++x) {
                if (edges.at(x, y) == 0) continue;
                votePixel(x, y);
            }
        }
        // Get the theta and rho with votes above threshold and draw lines
        for (size_t r = 0; r < accumulator.size(); ++r) {
            for (size_t t = 0; t < accumulator[r].size(); ++t) {
                if (accumulator[r][t] >= config_.lineThreshold && isLocalMaximum(r, t)) {
                    float votes = accumulator[r][t];
                    double rho = rhos_[r];
                    double theta = thetas_rad_[t];
                    insertTopN(top, votes, rho, theta);
                }
            }
        }
        for (const auto& line : top) {
            if (line.votes <= 0) continue;
            drawLines(lines, line.rho, line.theta);
        }
    }
    // Check if there are 
    bool checkLineValidity(size_t r_idx, size_t t_idx) {
        // Placeholder for line validity check, e.g., based on slope or position
        return true;
    }

    bool isLocalMaximum(size_t r_idx, size_t t_idx) {
        float current_value = accumulator[r_idx][t_idx];
        for (int dr = -3; dr <= 3; ++dr) {
            for (int dt = -3; dt <= 3; ++dt) {
                if (dr == 0 && dt == 0) continue;
                size_t neighbor_r = r_idx + dr;
                size_t neighbor_t = t_idx + dt;
                if (neighbor_r < 0 || neighbor_r >= accumulator.size() ||
                    neighbor_t < 0 || neighbor_t >= accumulator[0].size()) {
                    continue;
                }
                if (accumulator[neighbor_r][neighbor_t] >= current_value) {
                    return false;
                }
            }
        }
        return true;
    }

    void drawLines(Frame& lines, double rho, double theta) {
        // set the pixels in lines frame corresponding to the line defined by (rho, theta) to 255
        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a * rho, y0 = b * rho;
        int L = std::max(lines.width, lines.height);
        int x1 = static_cast<int>(x0 + L * (-b));
        int y1 = static_cast<int>(y0 + L * (a));
        int x2 = static_cast<int>(x0 - L * (-b));
        int y2 = static_cast<int>(y0 - L * (a));

        // Draw the line on the lines frame
        drawLine(lines, x1, y1, x2, y2);
    }

    void drawLine(Frame& frame, int x0, int y0, int x1, int y1) {
        const int ymin =ImageMask::getMaskStartY(frame.height); 
        cv::Point p0(x0, y0), p1(x1, y1);

        // Clip to image bounds; if no intersection, nothing to draw
        if (!cv::clipLine(cv::Size(frame.width, frame.height), p0, p1)) {
            return;
        }
        x0 = p0.x; y0 = p0.y;
        x1 = p1.x; y1 = p1.y;

        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2;

        while(true)
        {
            if (y0 >= ymin ) {
                frame.at(x0, y0) = 255;
            }
            if (x0 == x1 && y0 == y1) break;
            e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }

    static void insertTopN(std::vector<HoughLine>& top, float votes, double rho, double theta) {
        if (votes <= top.back().votes) return;
        top.back() = HoughLine{votes, rho, theta};
        std::sort(top.begin(), top.end(), [](const HoughLine& a, const HoughLine& b) {
            return a.votes > b.votes;
        });
    }

    const std::vector<HoughLine>& getDetectedLines() const override {
        return top;
    }
};
std::unique_ptr<HoughTransform> createHoughTransform() {
    return std::make_unique<hough_transform_impl>();
}