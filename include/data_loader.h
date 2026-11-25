#ifndef DATA_LOADER_H
#define DATA_LOADER_H
#include <string>

bool loadData(const std::string& filename, float* data, int count);
bool loadLabels(const std::string& filename, int* labels,int count);

#endif
