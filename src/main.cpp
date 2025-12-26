#include <iostream>
#include "video_service.h"
#include "canny_edge_detection.h"
#include <memory>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>


class VideoPipeline {
public:
    VideoPipeline(std::unique_ptr<VideoService> video_service, std::unique_ptr<CannyEdgeDetection> canny_edge_detector)
        : video_service_(std::move(video_service)), canny_edge_detector_(std::move(canny_edge_detector)) {}

    void run(const std::string& video_path) {
        if (!video_service_->initialize(video_path)) {
            std::cerr << "Failed to initialize video service." << std::endl;
            return;
        }
        cv::Mat cv_frame;
        Frame frame(0,0);
        while (video_service_->hasMoreFrames()) {
            cv_frame = video_service_->getFrame();
            frame = Frame::fromMat(cv_frame);
            processFrame(frame);
        }

        video_service_->releaseFrame(cv_frame);
    }

private:
    void processFrame(const Frame& frame) {
        // Show data
        Frame edges = canny_edge_detector_->run(frame);
        cv::Mat cv_frame = Frame::toMat(edges);
        cv::imshow("Canny Edges", cv_frame);
        cv::waitKey(30);
    }

    std::unique_ptr<VideoService> video_service_;
    std::unique_ptr<CannyEdgeDetection> canny_edge_detector_;
};


int main () {
    YAML::Node config = YAML::LoadFile("../config/main.yaml");
    std::string video_path = config["video_file"].as<std::string>();
    

    auto video_service = createVideoService();
    auto canny_edge_detector = createCannyEdgeDetection();
    VideoPipeline pipeline(std::move(video_service), std::move(canny_edge_detector));
    pipeline.run(video_path);
    return 0;
}