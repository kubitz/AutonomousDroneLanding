//
// Created by kubitz on 16/05/2021.
//

#include "../include/ObjectDetector.h"

ObjectDetector::ObjectDetector(const YoloConfig &net_config) {
    this->net_config = net_config;
    utils::iterate_file(net_config.path_labels, [&](const std::string &str) {
        this->labels.emplace_back(str);
    });
    this->net = dnn::readNetFromDarknet(net_config.path_cfg, net_config.path_weights);
    this->net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(dnn::DNN_TARGET_CPU);
}

Obstacles ObjectDetector::detect(cv::Mat &frame) {
    cv::Mat blob;
    dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(this->net_config.width_in, this->net_config.height_in),
                       cv::Scalar(0, 0, 0),
                       true, false);
    this->net.setInput(blob);
    std::vector<cv::Mat> boxes_out;
    this->net.forward(boxes_out, this->net.getUnconnectedOutLayersNames());
    Obstacles obstacles = this->postprocess(frame,boxes_out);
    double t_inf = this-> get_inference_time();
    std::string debug_info = cv::format("Inference time: %.2f ms", t_inf);
    std::cout << debug_info << std::endl;
    return obstacles;
}
Obstacles ObjectDetector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (auto &out:outs)
    {
        float* data = (float*)out.data;
        for (int j = 0; j < out.rows; ++j, data += out.cols)
        {
            cv::Mat scores = out.row(j).colRange(5, out.cols);
            cv::Point class_id_point;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
            if (confidence > this->net_config.conf_threshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                class_ids.emplace_back(class_id_point.x);
                confidences.emplace_back((float)confidence);
                boxes.emplace_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, this->net_config.conf_threshold, this->net_config.nms_threshold, indices);
    Obstacles obstacles;
    for (const auto& idx:indices)
    {
        Obstacle obstacle;
        obstacle.confidence = confidences[idx];
        obstacle.id = this->labels[class_ids[idx]];
        obstacle.box = boxes[idx];
        obstacles.emplace_back(obstacle);
        cv::Rect box = boxes[idx];
        this->draw_prediction(class_ids[idx], confidences[idx], box.x, box.y,
                              box.x + box.width, box.y + box.height, frame);
    }
    return obstacles;
}

void ObjectDetector::draw_prediction(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);
    std::string label = cv::format("%.2f", conf);
    if (!this->labels.empty())
    {
        CV_Assert(classId < (int)this->labels.size());
        label = this->labels[classId] + ":" + label;
    }
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
}

double ObjectDetector::get_inference_time() {
    std::vector<double> layers_inf_times;
    double freq = cv::getTickFrequency() /1000;
    return this->net.getPerfProfile(layers_inf_times) / freq;
}