#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../include/labels.h"
#include "../include/LzFinder.h"
#include <filesystem>
#include "../include/utils.h"
namespace fs = std::filesystem;
bool SIMULATE=true;


int main(int /*argc*/, char ** /*argv*/ ) {
    std::cout << labels_graz[0] << std::endl;
    LzFinder lz_finder("test");
    obstacle testObstacle{500, 500, 70, "human"};
    std::vector<obstacle> obstacles{testObstacle};

    std::cout << "Current path is " << fs::current_path() << '\n'; // (1)
    fs::current_path(fs::temp_directory_path()); // (3)
    std::cout << "Current path is " << fs::current_path() << '\n';

    std::vector<fs::path> imgs_to_process;
    std::vector<fs::path> masks_to_process;
    bool resImg = utils::iterate_file("/home/kubitz/CLionProjects/FYPLanding/data/cfg/inputs.cfg",
                                      [&](const std::string &str) {
                                          imgs_to_process.push_back(fs::path(str));
                                      });
    bool resMask = utils::iterate_file("/home/kubitz/CLionProjects/FYPLanding/data/cfg/masks.cfg",
                                      [&](const std::string &str) {
                                          imgs_to_process.push_back(fs::path(str));
                                      });
    if (resImg){
        int i = 0;
        i = 1;
    }

    cv::Mat img = cv::imread("/home/kubitz/CLionProjects/FYPLanding/data/imgs/0_simulation/images/041007_017.jpg", cv::IMREAD_COLOR);
    cv::Mat seg_img = cv::imread("/home/kubitz/CLionProjects/FYPLanding/data/imgs/0_simulation/masks/041007_017_mask.jpg", cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cout << "Could not read the image: " << std::endl;
        return 1;
    }
    if (seg_img.empty()) {
        std::cout << "Could not read the image: " << std::endl;
        return 1;
    }
    std::vector<landingZone> lzs_proposed = lz_finder.get_landing_zone_proposals(obstacles, 100, 100, img, "test");
    cv::Mat annotated_img = lz_finder.draw_lzs(img, lzs_proposed, obstacles);
    cv::Mat risk_map = lz_finder.get_risk_map(seg_img);
    lz_finder.rank_lzs(lzs_proposed, risk_map);
    fs::path test = utils::get_path_results(imgs_to_process[0]);
    fs::create_directories(test);
    utils::save_lzs(lzs_proposed,test / "lzs.csv");
    cv::applyColorMap(risk_map, risk_map, cv::COLORMAP_JET);
    cv::imshow("Display window", img);
    cv::imshow("risk_map", risk_map);
    cv::waitKey(0); // Wait for a keystroke in the window
    return 0;
}