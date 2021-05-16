//
// Created by kubitz on 12/05/2021.
//

#include "../include/utils.h"

bool utils::iterate_file(std::string fileName, std::function<void(const std::string &)> callback) {
    std::ifstream in(fileName.c_str());
    if (!in) {
        std::cerr << "Cannot open the File : " << fileName << std::endl;
        return false;
    }
    std::string str;
    while (std::getline(in, str)) {
        callback(str);
    }
    in.close();
    return true;
}

void utils::save_lzs(const std::vector<landingZone> &lzs, const std::string &path) {
    std::ofstream fout(path);
    fout << "confidence,radius,position,id" << std::endl;
    for (const auto &lz:lzs) {
        fout << lz.confidence << ",";
        fout << lz.radius << ",";
        fout << "\"(" << lz.posX << "," << lz.posY << ")\"" << ",";
        fout << lz.id << std::endl;
    }
    fout.close();
}

fs::path utils::get_path_results(fs::path img_path) {
    std::string seq_name;
    fs::path path_results;
    for (auto i = 0; i < 4; i++) {
        img_path = img_path.parent_path();
        if (i == 1)
            seq_name = img_path.filename();
    }
    path_results = img_path / "results";
    path_results /= seq_name;
    return path_results;
}

fs::path utils::get_data_path(fs::path exec_path) {
    fs::path data_path = exec_path.parent_path();
    data_path /= "data";
    return data_path;
}