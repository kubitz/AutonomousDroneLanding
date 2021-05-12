//
// Created by kubitz on 12/05/2021.
//

#ifndef FYPLANDING_LABELS_H
#define FYPLANDING_LABELS_H

#include "iostream"
#include "array"
#include "string"
#include <unordered_map>

using namespace std;
enum RiskLevel {
    ZERO = 0,
    LOW = 10,
    MEDIUM = 20,
    HIGH = 40,
    VERY_HIGH = 150
};

extern unordered_map<std::string, RiskLevel> seg_classes;
extern array<RiskLevel, 24> labels_graz;


#endif //FYPLANDING_LABELS_H
