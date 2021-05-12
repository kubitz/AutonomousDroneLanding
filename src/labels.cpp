//
// Created by kubitz on 12/05/2021.
//

#include "../include/labels.h"

unordered_map<std::string, RiskLevel> seg_classes({
                                                          {"unlabeled",     RiskLevel::ZERO},
                                                          {"pavedArea",     RiskLevel::LOW},
                                                          {"dirt",          RiskLevel::ZERO},
                                                          {"gravel",        RiskLevel::LOW},
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

array<RiskLevel, 24> labels_graz{
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

