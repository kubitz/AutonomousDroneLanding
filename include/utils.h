//
// Created by kubitz on 12/05/2021.
//

#ifndef FYPLANDING_UTILS_H
#define FYPLANDING_UTILS_H

#include "iostream"
#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include "../include/LzFinder.h"
#include "experimental/filesystem"

namespace fs = std::experimental::filesystem;

namespace utils {
    bool iterate_file(std::string fileName, std::function<void(const std::string &)> callback);

    void save_lzs(const std::vector<landingZone> &lzs, const std::string &path);

    fs::path get_path_results(fs::path img_path);
}


#endif //FYPLANDING_UTILS_H
