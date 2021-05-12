//
// Created by kubitz on 12/05/2021.
//

#ifndef FYPLANDING_LZFINDER_H
#define FYPLANDING_LZFINDER_H

typedef struct{
    int posX;
    int posY;
    int radius;
    string id;
    float confidence;
}landingZone;

typedef struct{
    int posX;
    int posY;
    int safety_radius;
    string id;
}obstacle;


class LzFinder {
public:
    string dataset;
    explicit LzFinder(const string& dataset);
    static Mat draw_lzs(const Mat &img, const vector<landingZone>& proposed_lzs, const vector<obstacle>& obstacles);
    vector<landingZone> get_landing_zone_proposals(const vector<obstacle>& obstacles, const int& stride, const int& r_landing,const Mat& img,const string& id);
    Mat get_risk_map(Mat const&  seg_img, int gaussian_sigma=250);
    void rank_lzs(vector<landingZone>& lzs,const Mat& risk_map,float weight_dist=3.0,float weight_risk=15.0);
    static circles_intersect(float x1,float x2, float y1, float y2, float r1, float r2);

private:
    void check_safety_requirements(landingZone & proposed_lzs, const vector<obstacle>& obstacles);
    static double get_norm_dist(const Mat& img,const Point& coordinates);
    double eval_risk_lz(const landingZone& lz, Mat risk_map);

    static bool compare_by_confidence(const landingZone &a, const landingZone &b);
};

#endif //FYPLANDING_LZFINDER_H
