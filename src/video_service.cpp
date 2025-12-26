#include "video_service.h"


struct VideoServiceImpl : public VideoService {

    cv::VideoCapture cap;
    bool initialized = false;

    cv::Mat getFrame() {

        cv::Mat frame;
        if (cap.read(frame)) {
            return frame;
        }
        return cv::Mat(); // Return an empty frame on failure
    }

    void releaseFrame(const cv::Mat& frame) {
        if (initialized) {
        }
    }

    bool initialize(const std::string& video_path) {
        cap.open(video_path);
        initialized = cap.isOpened();
        return initialized;
    }

    bool hasMoreFrames() {
        return initialized && cap.isOpened();
    }
};

std::unique_ptr<VideoService> createVideoService() {
    return std::make_unique<VideoServiceImpl>();
}