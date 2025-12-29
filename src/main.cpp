#include <iostream>
#include "video_service.h"
#include "canny_edge_detection.h"
#include "hough_transform.h"
#include <memory>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <chrono>

class VideoPipeline {
public:
    VideoPipeline(std::unique_ptr<VideoService> video_service, std::unique_ptr<CannyEdgeDetection> canny_edge_detector, std::unique_ptr<HoughTransform> hough_transform)
        : video_service_(std::move(video_service)), canny_edge_detector_(std::move(canny_edge_detector)), hough_transform_(std::move(hough_transform)) {}

    void run(const std::string& video_path) {
        if (!video_service_->initialize(video_path)) {
            std::cerr << "Failed to initialize video service." << std::endl;
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat cv_frame;
        Frame frame(0,0);
        int frame_count = 0;

        while (video_service_->hasMoreFrames()) {
            cv_frame = video_service_->getFrame();
            frame = Frame::fromMat(cv_frame);
            auto process_start = std::chrono::high_resolution_clock::now();
            processFrame(frame);
            auto process_end = std::chrono::high_resolution_clock::now();

            frame_count++;
            if (frame_count % 30 == 0) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                auto process_duration = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start);
                
                double fps = 30000.0 / total_duration.count();
                std::cout << "FPS: " << fps << ", Processing time: " << process_duration.count() << "ms" << std::endl;
                start_time = end_time;
            }
        }


        video_service_->releaseFrame(cv_frame);
    }

private:
    void processFrame(const Frame& frame) {
        // Show data
        Frame edges = canny_edge_detector_->run(frame);
        Frame lines = hough_transform_->run(edges);
        cv::Mat orig = Frame::toMat(frame);   // original frame
        cv::Mat lineImg = Frame::toMat(lines); // grayscale line image (0..255)

        // Ensure orig is 3-channel BGR for colored overlay
        cv::Mat origBgr;
        if (orig.channels() == 1) {
            cv::cvtColor(orig, origBgr, cv::COLOR_GRAY2BGR);
        } else {
            origBgr = orig.clone();
        }

        // Convert line image to BGR and colorize it (optional)
        cv::Mat linesBgr;
        cv::cvtColor(lineImg, linesBgr, cv::COLOR_GRAY2BGR);

        cv::Mat mask;
        cv::threshold(lineImg, mask, 1, 255, cv::THRESH_BINARY);
        cv::Mat redLines = cv::Mat::zeros(linesBgr.size(), linesBgr.type());
        redLines.setTo(cv::Scalar(0, 0, 255), mask); // BGR: red where mask is true

        // Blend
        cv::Mat overlay;
        cv::addWeighted(origBgr, 1.0, redLines, 1.0, 0.0, overlay);

        cv::imshow("Overlay (Hough lines)", overlay);
        cv::waitKey(1);
    }

    std::unique_ptr<VideoService> video_service_;
    std::unique_ptr<CannyEdgeDetection> canny_edge_detector_;
    std::unique_ptr<HoughTransform> hough_transform_;
};


int main () {
    YAML::Node config = YAML::LoadFile("../config/main.yaml");
    std::string video_path = config["video_file"].as<std::string>();
    

    auto video_service = createVideoService();
    auto canny_edge_detector = createCannyEdgeDetection();
    auto hough_transform = createHoughTransform();
    VideoPipeline pipeline(std::move(video_service), std::move(canny_edge_detector), std::move(hough_transform));
    pipeline.run(video_path);
    return 0;
}