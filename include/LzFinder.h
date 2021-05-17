//
// Created by kubitz on 12/05/2021.
//

#ifndef FYPLANDING_LZFINDER_H
#define FYPLANDING_LZFINDER_H

#include "iostream"
#include "opencv2/imgproc.hpp"
#include "string"
#include "cmath"
#include <algorithm>
#include "labels.h"


typedef struct {
    int posX;
    int posY;
    int radius;
    std::string id;
    float confidence;
} LandingZone;

typedef struct {
    cv::Rect box;
    int safety_radius;
    std::string id;
    float confidence;

    cv::Point get_center() const { return (box.br() + box.tl()) * 0.5; }
} Obstacle;

typedef std::vector<LandingZone> LandingZones;
typedef std::vector<Obstacle> Obstacles;

class LzFinder {
public:

    explicit LzFinder(const std::string &dataset);

    static cv::Mat
    draw_lzs(const cv::Mat &img, const std::vector<LandingZone> &proposed_lzs, const Obstacles &obstacles);

    std::vector<LandingZone>
    get_landing_zone_proposals(const Obstacles &obstacles, const int &stride, const int &r_landing,
                               const cv::Mat &img, const std::string &id);

    static int circles_intersect(float x1, float x2, float y1, float y2, float r1, float r2);

    cv::Mat get_risk_map(cv::Mat const &seg_img, int gaussian_sigma = 250);

    void
    rank_lzs(std::vector<LandingZone> &lzs, const cv::Mat &risk_map, float weight_dist = 3.0, float weight_risk = 15.0);

    std::vector<LandingZone>
    get_ranked_lzs(const cv::Mat &seg_img, Obstacles obstacles, const int &stride, const std::string &id,
                   const int &r_landing, const int &gaussian_sigma = 255);

private:
    void check_safety_requirements(LandingZone &proposed_lzs, const Obstacles &obstacles);

    static double get_norm_dist(const cv::Mat &img, const cv::Point &coordinates);

    double eval_risk_lz(const LandingZone &lz, const cv::Mat &risk_map);

    static bool compare_by_confidence(const LandingZone &a, const LandingZone &b);

};

#endif //FYPLANDING_LZFINDER_H
