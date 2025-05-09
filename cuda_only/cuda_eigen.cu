#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>

using namespace std;

/* cuda error check func for cudaMalloc and cudaMemcpy */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

/* cuslver error check func */
#define cusolverErrchk(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t status, const char *file, int line)
{
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr,"cuSolver Error at %s:%d code=%d\n", file, line, status);
        exit(status);
    }
}

/* normalize the input adjacency matrix*/
void normalize_adjacency(float* h_graph, int n){
    vector<float> deg(n, 0.0f);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            /*TODO: access element at (i, j) in column-major layout */
            deg[i] += h_graph[i + j*n];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (deg[i] > 0 && deg[j] > 0)
                h_graph[i + j*n] = h_graph[i + j*n] / sqrt(deg[i]*deg[j]);
            else
                /* handle divide-by-zero case */
                h_graph[i + j*n] = 0.0f;
        }
    }
}

/* build the normalized laplacian matrix L = I - A_norm */
void build_laplacian(float* h_L, float* h_graph, int n){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; j++) {
            if(i == j)
                /*REMEMBER: diagonal and off-diagonal cases*/
                h_L[i + j*n] = 1.0f - h_graph[i + j*n];
            else
                h_L[i + j*n] = 0.0f - h_graph[i + j*n];
        }
    }
}

/* standradize the sign of the eigenvectors to match the serial Eigen libaray*/
void standardize_sign(float* V, int n, int k) {
    for (int j = 0; j < k; j++) {
        float max_val = 0.0f;
        for (int i = 0; i < n; i++) {
            /*TODO: access element (i, j) in column-major format */
            float v = V[i + j*n];
            if (fabs(v) > fabs(max_val))
                max_val = v;
        }
        /* if the dominant entry is negative, flip the sign of the whole vector*/
        if (max_val < 0.0f) {
            for (int i = 0; i < n; i++)
                V[i + j*n] = -V[i + j*n];
        }
    }
}

void eigen_computation_cuda(float* h_graph, int n, int k, vector<vector<float>>& out){
    /* normalize the input adjacency matrix; call the norm func*/
    float* h_norm = (float*)malloc(n*n*sizeof(float));
    memcpy(h_norm, h_graph, n*n*sizeof(float));
    normalize_adjacency(h_norm, n);

    /* laplacian serial */
    float* h_L = (float*)malloc(n*n*sizeof(float));
    build_laplacian(h_L, h_norm, n);

    /* move the laplacian matrix to gpu */
    float *d_L = nullptr;
    gpuErrchk(cudaMalloc(&d_L, n*n*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_L, h_L, n*n*sizeof(float), cudaMemcpyHostToDevice));

    float* d_W = nullptr;
    gpuErrchk(cudaMalloc(&d_W, n*sizeof(float)));
    int* devInfo = nullptr;
    gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));

    cusolverDnHandle_t handle = nullptr;
    cusolverErrchk(cusolverDnCreate(&handle));

    /* buffer memory allocation on gpu for the solver func to compute the eigen values/vectors*/
    int l_work = 0;
    cusolverErrchk(cusolverDnSsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, d_L, n, d_W, &l_work));
    
    float* d_work = nullptr;
    gpuErrchk(cudaMalloc(&d_work, l_work*sizeof(float)));

    /* actual eigen value solver function from the cusolver library*/
    /* get both the eigen vectors and values here*/
    cusolverErrchk(cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, d_L, n, d_W, d_work, l_work, devInfo));

    /* copy eigenvectors back to cpu for clustering rn*/
    float* h_V = (float*)malloc(n*n*sizeof(float));
    gpuErrchk(cudaMemcpy(h_V, d_L, n*n*sizeof(float), cudaMemcpyDeviceToHost));

    /* normalize signs for consistency */
    int k_true = min(k, n);
    standardize_sign(h_V, n, k_true);

    out.assign(n, vector<float>(k_true));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k_true; j++)
            out[i][j] = h_V[i + j*n];

    /* free up stuff*/
    free(h_norm);
    free(h_L);
    free(h_V);
    cudaFree(d_L);
    cudaFree(d_W);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);
}