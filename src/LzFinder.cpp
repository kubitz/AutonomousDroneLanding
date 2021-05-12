//
// Created by kubitz on 12/05/2021.
//

#include "../include/LzFinder.h"

LzFinder::LzFinder(const string &dataset) {
};

int LzFinder::circles_intersect(float x1, float x2, float y1, float y2, float r1, float r2) {
    auto d = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
    if (d < (r1 - r2)) {
        // C2 is in C1
        return -3;
    }
    if (d < (r2 - r1)) {
        // C1 is in C2
        return -2;
    }
    if (d > (r1 + r2)) {
        // Circumference of C1 and C2 overlap
        return 0;
    } else {
        // Circles do not overlap
        return -1;
    }
}

vector<landingZone> LzFinder::get_landing_zone_proposals(const vector<obstacle> &obstacles, const int &stride,
                                                         const int &r_landing, const Mat &img, const string &id) {
    vector<landingZone> proposed_lzs;
    for (auto y = 0; y < img.rows; y += stride) {
        for (auto x = 0; x < img.cols; x += stride) {
            landingZone lz = {x, y, r_landing, id, NAN};
            this->check_safety_requirements(lz, obstacles);
            proposed_lzs.push_back(lz);
        }
    }
    return proposed_lzs;
}

Mat LzFinder::draw_lzs(const Mat &img, const vector<landingZone> &proposed_lzs, const vector<obstacle> &obstacles) {
    Mat img_annotated = img;
    for (const auto &lz: proposed_lzs) {
        circle(img_annotated, Point(lz.posX, lz.posY), lz.radius, Scalar(0, 255, 0), 1, -1);
    }
    for (const auto &obstacle: obstacles) {
        circle(img_annotated, Point(obstacle.posX, obstacle.posY), obstacle.safety_radius, Scalar(0, 0, 255), 1, -1);
    }
    return img_annotated;
}

void LzFinder::check_safety_requirements(landingZone &lz, const vector<obstacle> &obstacles) {
    for (const auto &ob : obstacles) {
        int touch = LzFinder::circles_intersect(lz.posX, ob.posX, lz.posY, ob.posY, lz.radius, ob.safety_radius);
        if (touch < 0) {
            lz.confidence = 0;
        }
    }
}

double LzFinder::get_norm_dist(const Mat &img, const Point &coordinates) {
    double furthestDist = hypot(img.cols / 2, img.rows / 2);
    Point center{img.cols / 2, img.rows / 2};
    Point2f diff = coordinates - center;
    double dist = cv::sqrt(diff.x * diff.x + diff.y * diff.y);
    double norm_dist = 1 - abs(dist / furthestDist);
    return norm_dist;
}

Mat LzFinder::get_risk_map(Mat const &seg_img, int gaussian_sigma) {
    Mat risk_map;
    if (seg_img.depth() != 1) {
        vector<Mat> channels;
        split(seg_img, channels);
        risk_map = channels[0];
    } else {
        risk_map = seg_img.clone();
    }
    for (auto i = 0; i < labels_graz.size(); i++) {
        // TODO: use cv::LUT() method for faster implementation
        risk_map.setTo(labels_graz[i], risk_map == i);
    }
    GaussianBlur(risk_map, risk_map, Size(51, 51), gaussian_sigma);
    normalize(risk_map, risk_map, 0, 255, cv::NORM_MINMAX);
    return risk_map;
}

double LzFinder::eval_risk_lz(const landingZone &lz, const Mat &risk_map) {
    Mat crop;
    Mat mask(risk_map.rows, risk_map.cols, CV_8UC1, Scalar(0));
    circle(mask, Point(lz.posX, lz.posY), lz.radius, Scalar(255), -1);
    bitwise_and(risk_map, mask, crop);
    double risk = sum(crop)[0];
    double areaLz = M_PI * pow(lz.radius, 2);
    return risk / (areaLz * 255);
}

void LzFinder::rank_lzs(vector<landingZone> &lzs, const Mat &risk_map, float weight_dist, float weight_risk) {

    for (auto &lz: lzs) {
        if (isnan(lz.confidence)) {
            double risk = this->eval_risk_lz(lz, risk_map);
            double dist = this->get_norm_dist(risk_map, Point(lz.posX, lz.posY));
            lz.confidence = float(weight_dist * dist + weight_risk * risk) / (weight_dist + weight_risk);
        }
    }
    sort(lzs.begin(), lzs.end(), LzFinder::compare_by_confidence);
}


bool LzFinder::compare_by_confidence(const landingZone &a, const landingZone &b) {
    return a.confidence < b.confidence;
}