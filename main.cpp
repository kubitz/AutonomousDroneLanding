#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "string"
#include <iostream>
#include "cmath"
#include <algorithm>

const bool SIMULATE = true;
using namespace cv;
using namespace std;

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

enum RiskLevel{
    ZERO=0,
    LOW=10,
    MEDIUM=20,
    HIGH=40,
    VERY_HIGH=150
};

unordered_map <std::string, RiskLevel> seg_classes({
                                  {"unlabeled",     RiskLevel::ZERO},
                                  {"pavedArea",     RiskLevel::LOW},
                                  {"dirt",          RiskLevel::ZERO},
                                  {"gravel",          RiskLevel::LOW},
                                  {"grass",         RiskLevel::ZERO},
                                  {"water",         RiskLevel::HIGH},
                                  {"rocks",         RiskLevel::MEDIUM},
                                  {"pool",          RiskLevel::HIGH},
                                  {"lowVegetation", RiskLevel::ZERO},
                                  {"roof",          RiskLevel::HIGH},
                                  {"wall",          RiskLevel::HIGH},
                                  {"window",        RiskLevel::HIGH},
                                  {"door",          RiskLevel::HIGH},
                                  {"fence",         RiskLevel::HIGH},
                                  {"fencePole",     RiskLevel::HIGH},
                                  {"person",        RiskLevel::VERY_HIGH},
                                  {"animal",        RiskLevel::VERY_HIGH},
                                  {"car",           RiskLevel::VERY_HIGH},
                                  {"bike",          RiskLevel::VERY_HIGH},
                                  {"tree",          RiskLevel::HIGH},
                                  {"baldTree",      RiskLevel::HIGH},
                                  {"arMarker",      RiskLevel::ZERO},
                                  {"obstacle",      RiskLevel::HIGH},
                                  {"conflicting",   RiskLevel::HIGH},
                                  {"background",    RiskLevel::ZERO},
                                  {"drone",         RiskLevel::MEDIUM},
                                  {"boat",          RiskLevel::MEDIUM},
                                  {"construction",  RiskLevel::HIGH},
                                  {"vegetation",    RiskLevel::LOW},
                                  {"road",          RiskLevel::LOW},
                                  {"sky",           RiskLevel::VERY_HIGH}
                          });

array <RiskLevel,24> labels {
    seg_classes.at("unlabeled"),
    seg_classes.at("pavedArea"),
    seg_classes.at("dirt"),
    seg_classes.at("grass"),
    seg_classes.at("gravel"),
    seg_classes.at("water"),
    seg_classes.at("rocks"),
    seg_classes.at("pool"),
    seg_classes.at("lowVegetation"),
    seg_classes.at("roof"),
    seg_classes.at("wall"),
    seg_classes.at("window"),
    seg_classes.at("door"),
    seg_classes.at("fence"),
    seg_classes.at("fencePole"),
    seg_classes.at("person"),
    seg_classes.at("animal"),
    seg_classes.at("car"),
    seg_classes.at("bike"),
    seg_classes.at("tree"),
    seg_classes.at("baldTree"),
    seg_classes.at("arMarker"),
    seg_classes.at("obstacle"),
    seg_classes.at("conflicting")
};


class LzFinder {
public:
    string dataset;
    LzFinder(const string& dataset);
    static Mat draw_lzs(const Mat& img, vector<landingZone> proposed_lzs);
    vector<landingZone> get_landing_zone_proposals(const vector<obstacle>& obstacles, const int& stride, const int& r_landing,const Mat& img,const string& id);
    Mat get_risk_map(Mat const&  seg_img, int gaussian_sigma=250);
    void rank_lzs(vector<landingZone>& lzs,const Mat& risk_map,float weight_dist=3.0,float weight_risk=15.0);

private:
    int circles_intersect(float x1,float x2, float y1, float y2, float r1, float r2);
    void check_safety_requirements(landingZone & proposed_lzs, const vector<obstacle>& obstacles);
    static double get_norm_dist(const Mat& img,const Point& coordinates);
    double eval_risk_lz(const landingZone& lz, Mat risk_map);

    static bool compare_by_confidence(const landingZone &a, const landingZone &b);
};

LzFinder::LzFinder(const string& dataset) {
    this->dataset=dataset;
};
int LzFinder::circles_intersect(float x1, float x2, float y1, float y2, float r1, float r2) {
    auto d = sqrt(pow((x1 - x2),2) + pow((y1 - y2),2));
    if (d < (r1 - r2)){
        // C2 is in C1
        return -3;
    }
    if (d < (r2-r1)){
        // C1 is in C2
        return -2;
    }
    if (d>(r1+r2)){
        // Circumference of C1 and C2 overlap
        return 0;
    }
    else {
        // Circles do not overlap
        return -1;
    }
}
vector<landingZone > LzFinder::get_landing_zone_proposals(const vector<obstacle> &obstacles, const int &stride,
                                                          const int &r_landing, const Mat &img, const string &id) {
    vector<landingZone> proposed_lzs;
    for (auto y=0;y<img.rows;y+=stride){
        for (auto x=0;x<img.cols;x+=stride){
            landingZone lz ={ x, y, r_landing, id, NAN};
            this->check_safety_requirements(lz,obstacles);
            proposed_lzs.push_back(lz);
        }
    }
    return proposed_lzs;
}
Mat LzFinder::draw_lzs(const Mat &img, vector<landingZone> proposed_lzs) {
    Mat img_annotated = img;
    for(landingZone lz: proposed_lzs){
        if (lz.confidence==0){
            circle( img_annotated, Point( lz.posX, lz.posY ), lz.radius, Scalar( 0, 0, 255 ), 1, -1 );
        }
        else{
            circle( img_annotated, Point( lz.posX, lz.posY ), lz.radius, Scalar( 0, 255, 0 ), 1, -1 );
        }
    }
    return img_annotated;
}

void LzFinder::check_safety_requirements(landingZone &lz, const vector<obstacle> &obstacles) {
    for(auto ob : obstacles)
    {
        int touch = LzFinder::circles_intersect(lz.posX,ob.posX,lz.posY,ob.posY,lz.radius,ob.safety_radius);
        if (touch<0){
            lz.confidence=0;
        }
    }
}

double LzFinder::get_norm_dist(const Mat &img, const Point& coordinates) {
    double furthestDist = hypot(img.cols/2,img.rows/2);
    Point center{img.cols/2,img.rows/2};
    Point2f diff = coordinates - center;
    double dist = cv::sqrt(diff.x*diff.x + diff.y*diff.y);
    double norm_dist = 1- abs(dist/furthestDist);
    return norm_dist;
}

Mat LzFinder::get_risk_map(Mat const& seg_img, int gaussian_sigma) {
    Mat risk_map;
    if(seg_img.depth()!=1){
        vector<Mat> channels;
        split(seg_img,channels);
        risk_map = channels[0];
    }
    else{
        risk_map = seg_img.clone();
    }
    for (auto i=0; i<labels.size(); i++) {
        // TODO: use cv::LUT() method for faster implementation
        risk_map.setTo(labels[i],risk_map==i);
    }
    GaussianBlur(risk_map, risk_map, Size(51, 51), gaussian_sigma);
    normalize(risk_map, risk_map, 0, 255, cv::NORM_MINMAX);
    return risk_map;
}

double LzFinder::eval_risk_lz(const landingZone &lz, Mat risk_map) {
    Mat crop;
    Mat mask(risk_map.rows, risk_map.cols, CV_8UC1, Scalar(0));
    circle(mask,Point(lz.posX,lz.posY),lz.radius,Scalar(255),-1);
    bitwise_and(risk_map,mask,crop);
    double risk = sum(crop)[0];
    double areaLz = M_PI*pow(lz.radius,2);
    return risk/(areaLz*255);
}

void LzFinder::rank_lzs(vector<landingZone>& lzs, const Mat& risk_map, float weight_dist, float weight_risk) {

    for (auto& lz: lzs){
        if (isnan(lz.confidence)){
            double risk = this->eval_risk_lz(lz,risk_map);
            double dist = this->get_norm_dist(risk_map,Point(lz.posX,lz.posY));
            lz.confidence = float (weight_dist*dist+weight_risk*risk)/(weight_dist+weight_risk);
        }
    }
    sort(lzs.begin(),lzs.end(),this->compare_by_confidence);
}

bool LzFinder::compare_by_confidence(const landingZone& a, const landingZone& b)
{
    return a.confidence < b.confidence;
}

int main( int /*argc*/, char** /*argv*/ )
{
    cout<< labels[0] << endl;
    LzFinder lz_finder("test");
    obstacle testObstacle{500,500,70,"human"};
    vector<obstacle > obstacles{testObstacle};

    std::string image_path = samples::findFile("/home/kubitz/CLionProjects/FYPLanding/data/images/041007_017.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    Mat seg_img = imread("/home/kubitz/CLionProjects/FYPLanding/data/masks/041007_017_mask.jpg", IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    if(seg_img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    vector<landingZone > lzs_proposed = lz_finder.get_landing_zone_proposals(obstacles,100,100,img,"test");
    Mat annotated_img = lz_finder.draw_lzs(img,lzs_proposed);
    Mat risk_map=lz_finder.get_risk_map(seg_img);
    lz_finder.rank_lzs(lzs_proposed,risk_map);
    applyColorMap(risk_map,risk_map,COLORMAP_JET);
    imshow("Display window", img);
    imshow("risk_map",risk_map);
    int k = waitKey(0); // Wait for a keystroke in the window



    return 0;
}