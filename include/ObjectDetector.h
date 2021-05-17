//
// Created by kubitz on 16/05/2021.
//

#ifndef FYPLANDING_OBJECTDETECTOR_H
#define FYPLANDING_OBJECTDETECTOR_H

#include "iostream"
#include "string"
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../include/utils.h"

namespace dnn = cv::dnn;

typedef struct {
    float conf_threshold;
    float nms_threshold; // Non-maximum suppression threshold
    int width_in;
    int height_in;
    std::string path_cfg;
    std::string path_weights;
    std::string path_labels;
} YoloConfig;

class ObjectDetector {
public:
    ObjectDetector(const YoloConfig &net_config);

    Obstacles detect(cv::Mat &frame);

private:
    YoloConfig net_config;
    dnn::Net net;
    std::vector<std::string> labels;

    Obstacles postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs);

    double get_inference_time();

    void draw_prediction(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

};


#endif //FYPLANDING_OBJECTDETECTOR_H
