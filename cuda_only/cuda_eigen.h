#ifndef CUDA_H
#define CUDA_H

#include <vector>
using namespace std;
/* header for the eigensolver func*/
void eigen_computation_cuda(float* h_graph, int n, int k, vector<vector<float>>& out);

#endif