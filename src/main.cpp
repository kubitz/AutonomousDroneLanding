#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../include/labels.h"
#include "../include/LzFinder.h"
#include <experimental/filesystem>
#include "algorithm"
#include "../include/utils.h"
#include "../include/ObjectDetector.h"
namespace fs = std::experimental::filesystem;
bool SIMULATE = false;


int main(int /*argc*/, char ** /*argv*/ ) {
    LzFinder lz_finder("test");
    std::vector<fs::path> imgs_to_process;
    fs::path cwd = fs::current_path();
    fs::path data_path = utils::get_data_path(cwd);
    auto resImg = utils::iterate_file(std::string(data_path / "cfg" / "inputs.cfg"),
                                      [&](const std::string &str) {
                                          imgs_to_process.emplace_back(fs::path(str));
                                      });
    Obstacle testObstacle{cv::Rect(cv::Point(500,500),cv::Point(450,450)), 70, "human", NAN};
    Obstacles obstacles{testObstacle};
    LandingZones lzs;
    if (SIMULATE) {
        std::vector<fs::path> masks_to_process;
        auto resMask = utils::iterate_file(data_path / "cfg" / "masks.cfg",
                                           [&](const std::string &str) {
                                               masks_to_process.emplace_back(fs::path(str));
                                           });
        for (auto i = 0; i < imgs_to_process.size(); i++) {
            cv::Mat mask = cv::imread(std::string(masks_to_process[i]),cv::IMREAD_COLOR);
            if (mask.empty()) {
                std::cout << "Could not read the segmentation mask: " << std::string(masks_to_process[i]) << std::endl;
                return 1;
            }
            std::string id = imgs_to_process[i].stem();
            LandingZones lzs_seq = lz_finder.get_ranked_lzs(mask, obstacles, 100,
                                                            id, 100, 250);
            std::move(lzs_seq.begin(), lzs_seq.end(), std::back_inserter(lzs));
        }
    } else {
        // TODO: Add semantic engine
        int i;
        YoloConfig obj_config{0, 0.3, 416, 416,
                          "/home/kubitz/CLionProjects/FYPLanding/data/weights/yolo-v3/yolov3_leaky.cfg",
                          "/home/kubitz/CLionProjects/FYPLanding/data/weights/yolo-v3/yolov3_leaky.weights",
                          "/home/kubitz/CLionProjects/FYPLanding/data/weights/yolo-v3/visdrone.names"
        };
        cv::Mat img = cv::imread(data_path / "imgs" / "seq3"/ "images" / "040001_043.jpg",
                                 cv::IMREAD_COLOR);
        ObjectDetector obj_net(obj_config);
        Obstacles obstacles = obj_net.detect(img);
        cv::imshow("Display window", img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    fs::path path_results = utils::get_path_results(imgs_to_process[0]);
    fs::create_directories(path_results);
    utils::save_lzs(lzs, path_results / "results_lzs.csv");


    /*
     * This section is just to test the individual functions - it will be removed in the final code
     */
    cv::Mat img = cv::imread(data_path / "imgs" / "0_simulation"/ "images" / "041007_017.jpg",
                             cv::IMREAD_COLOR);
    cv::Mat seg_img = cv::imread(
            data_path / "imgs"/ "0_simulation"/"masks"/"041007_017_mask.jpg", cv::IMREAD_COLOR);


    if (seg_img.empty()) {
        std::cout << "Could not read the image: " << std::endl;
        return 1;
    }
    LandingZones lzs_proposed = lz_finder.get_landing_zone_proposals(obstacles, 100, 100, img, "test");
    cv::Mat annotated_img = lz_finder.draw_lzs(img, lzs_proposed, obstacles);
    cv::Mat risk_map = lz_finder.get_risk_map(seg_img);
    lz_finder.rank_lzs(lzs_proposed, risk_map);
    cv::applyColorMap(risk_map, risk_map, cv::COLORMAP_JET);
    cv::imshow("Display window", img);
    cv::imshow("risk_map", risk_map);
    cv::waitKey(0); // Wait for a keystroke in the window
    return 0;
}