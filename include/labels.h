//
// Created by kubitz on 12/05/2021.
//

#ifndef FYPLANDING_LABELS_H
#define FYPLANDING_LABELS_H

#include "iostream"
#include "array"
#include "string"
#include <unordered_map>

enum RiskLevel {
    ZERO = 0,
    LOW = 10,
    MEDIUM = 20,
    HIGH = 40,
    VERY_HIGH = 150
};

extern std::unordered_map<std::string, RiskLevel> seg_classes;
extern std::array<RiskLevel, 24> labels_graz;

#endif //FYPLANDING_LABELS_H
