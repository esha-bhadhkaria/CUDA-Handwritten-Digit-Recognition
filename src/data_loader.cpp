#include <iostream>
#include <fstream>
#include "data_loader.h"

bool loadData(const std::string& filename, float* data, int count) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot open " << filename << "\n";
        return false;
    }

    file.read(reinterpret_cast<char*>(data), count * sizeof(float));
    return file.good() || file.eof();
}

bool loadLabels(const std::string& filename, int* labels, int count) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot open " << filename << "\n";
        return false;
    }

    file.read(reinterpret_cast<char*>(labels), count * sizeof(int));
    return file.good() || file.eof();
}
