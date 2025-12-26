#include "types.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>


struct VideoService {
    virtual ~VideoService() = default;
    virtual cv::Mat getFrame() = 0;
    virtual void releaseFrame(const cv::Mat& frame) = 0;
    virtual bool initialize(const std::string& video_path) = 0;
    virtual bool hasMoreFrames() = 0;
};

std::unique_ptr<VideoService> createVideoService();