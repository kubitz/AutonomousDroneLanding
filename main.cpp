#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "string"
#include <iostream>
#include "cmath"

const bool SIMULATE = True;
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
    VERY_HIGH=100
};

enum SegClasses{
    "unlabeled": RiskLevel.ZERO,
    "pavedArea": RiskLevel.LOW,
    "dirt": RiskLevel.LOW,
    "grass": RiskLevel.ZERO,
    "dirt": RiskLevel.LOW,
    "water": RiskLevel.HIGH,
    "rocks": RiskLevel.MEDIUM,
    "pool": RiskLevel.HIGH,
    "lowVegetation": RiskLevel.ZERO,
    "roof": RiskLevel.HIGH,
    "wall": RiskLevel.HIGH,
    "window": RiskLevel.HIGH,
    "door": RiskLevel.HIGH,
    "fence": RiskLevel.HIGH,
    "fencePole": RiskLevel.HIGH,
    "person": RiskLevel.VERY_HIGH,
    "animal": RiskLevel.VERY_HIGH,
    "car": RiskLevel.VERY_HIGH,
    "bike": RiskLevel.VERY_HIGH,
    "tree": RiskLevel.HIGH,
    "baldTree": RiskLevel.HIGH,
    "arMarker": RiskLevel.ZERO,
    "obstacle": RiskLevel.HIGH,
    "conflicting": RiskLevel.HIGH,
    "background": RiskLevel.ZERO,
    "drone": RiskLevel.MEDIUM,
    "boat": RiskLevel.MEDIUM,
    "construction": RiskLevel.HIGH,
    "vegetation": RiskLevel.LOW,
    "road": RiskLevel.ZERO,
    "sky": RiskLevel.VERY_HIGH,
};

class LzFinder {
public:
    string dataset;
    LzFinder(const string& dataset);
    Mat draw_lzs(const Mat& img, vector<landingZone> proposed_lzs);
    vector<landingZone> get_landing_zone_proposals(const vector<obstacle>& obstacles, const int& stride, const int& r_landing,const Mat& img,const string& id);
private:
    int circles_intersect(float x1,float x2, float y1, float y2, float r1, float r2);
    void check_safety_requirements(landingZone & proposed_lzs, const vector<obstacle>& obstacles);
    float get_norm_dist(const Mat& img,Point coordinates);
    Mat get_risk_map(Mat seg_img, int gaussian_sigma=25);
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

float LzFinder::get_norm_dist(const Mat &img, Point coordinates) {
    float furthestDist = hypot(img.cols/2,img.rows/2);
    Point center{img.cols/2,img.rows/2};
    Point2f diff = coordinates - center;
    float dist = cv::sqrt(diff.x*diff.x + diff.y*diff.y);
    float norm_dist = 1- abs(dist/furthestDist);
    return norm_dist;
}

Mat LzFinder::get_risk_map(Mat seg_img, int gaussian_sigma) {
    if(SIMULATE== true){
        cv::extractChannel(seg_img, seg_img, 0);
    }
    for (auto label:this->labels){
        cout << "Implement label finding thing" << endl;
    }
    /*
    risk_array = image.astype("float32")
    for label in self.labels:
    np.where(risk_array == self.labels[label], risk_table[label], risk_array)
    risk_array = gaussian_filter(risk_array, sigma=gaussian_sigma)
    risk_array = (risk_array / risk_array.max()) * 255
    risk_array = np.uint8(risk_array)
    return risk_array
    */
}

int main( int /*argc*/, char** /*argv*/ )
{
    LzFinder lz_finder("test");
    obstacle testObstacle{500,500,70,"human"};
    vector<obstacle > obstacles{testObstacle};

    std::string image_path = samples::findFile("/home/kubitz/CLionProjects/FYPLanding/data/images/041007_017.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    vector<landingZone > lzs_proposed = lz_finder.get_landing_zone_proposals(obstacles,100,100,img,"test");
    Mat annotated_img = lz_finder.draw_lzs(img,lzs_proposed);
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }
    return 0;
}