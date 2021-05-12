#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../include/labels.h"
#include "../include/LzFinder.h"

int main(int /*argc*/, char ** /*argv*/ ) {
    cout << labels_graz[0] << endl;
    LzFinder lz_finder("test");
    obstacle testObstacle{500, 500, 70, "human"};
    vector<obstacle> obstacles{testObstacle};

    std::string image_path = samples::findFile("/home/kubitz/CLionProjects/FYPLanding/data/images/041007_017.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    Mat seg_img = imread("/home/kubitz/CLionProjects/FYPLanding/data/masks/041007_017_mask.jpg", IMREAD_COLOR);

    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    if (seg_img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    vector<landingZone> lzs_proposed = lz_finder.get_landing_zone_proposals(obstacles, 100, 100, img, "test");
    Mat annotated_img = lz_finder.draw_lzs(img, lzs_proposed, obstacles);
    Mat risk_map = lz_finder.get_risk_map(seg_img);
    lz_finder.rank_lzs(lzs_proposed, risk_map);
    applyColorMap(risk_map, risk_map, COLORMAP_JET);
    imshow("Display window", img);
    imshow("risk_map", risk_map);
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}