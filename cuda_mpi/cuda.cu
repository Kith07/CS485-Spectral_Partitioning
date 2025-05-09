#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>

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

/* Calculate the adjacency matrix from the given input graph/nodes
void adj_matrix(FILE *file, float** graph, int n, float d, int m, int idx) {
    int src, dest;

    for (int i = 0; i < n; i++) {
        graph[i] = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; ++j) {
            graph[i][j] = (1 - d) / float(n);
        }
    }

    while (m--) {
        fscanf(file, "%d%d", &src, &dest);
        if (idx == 0)
            graph[dest][src] += d * 1.0;
        else
            graph[dest - 1][src - 1] += d * 1.0;
    }
}
*/

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
void build_laplacian(float* h_L, const float* h_graph, int n){
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

/* Move the main function into the MPI program for parallel clustering
int main(int argc, char** argv) {
    clock_t start, end;
    FILE* file;
    int n, m, idx;
    int count = 10, k = 10, node = -1;
    float d = 0.85;

    if (argc > 2)
        node = atoi(argv[2]);

    file = fopen(argv[1], "r");
    fscanf(file, "%d %d %d", &n, &m, &idx);

    // create adjacency matrix in main instead
    
    float** graph = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        graph[i] = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; ++j)
            graph[i][j] = (1 - d) / float(n);
    }

    int src, dest;
    while (m--) {
        fscanf(file, "%d %d", &src, &dest);
        if (idx == 0)
            graph[dest][src] += d * 1.0f;
        else
            graph[dest - 1][src - 1] += d * 1.0f;
    }
    fclose(file);

    // flatten the graph to 1d array
    float* flat_graph = (float*)malloc(n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            flat_graph[i * n + j] = graph[i][j];

    // call the cuda solver func
    vector<vector<float>> eigen_matrix;
    start = clock();
    vector<vector<int>> clusters = spectral_partition_cuda(flat_graph, n, k, eigen_matrix);
    end = clock();
    // TODO: remember to MPI use the k means clustering
    if (node >= 0 && node < n) {
        vector<pair<float, int>> dists;
        vector<float> query = eigen_matrix[node];

        for (int i = 0; i < n; i++) {
            if (i == node) continue;
            float dist = 0.0f;
            for (int j = 0; j < query.size(); ++j)
                dist += (query[j] - eigen_matrix[i][j]) * (query[j] - eigen_matrix[i][j]);
            dist = sqrt(dist);
            dists.push_back({dist, i});
        }

        sort(dists.begin(), dists.end());
        printf("Top %d Nearest Nodes/Pages of Node %d (Eigen Space):\n", count, node);
        for (int i = 0; i < count && i < dists.size(); ++i) {
            printf("%2d. Node %d (Euclidean Distance = %.6f)\n", i + 1, dists[i].second, dists[i].first);
        }
    }

    printf("\nTime for %d nodes: %.6f seconds\n", n, float(end - start) / CLOCKS_PER_SEC);

    for (int i = 0; i < n; i++)
        free(graph[i]);
    free(graph);
    free(flat_graph);

    return 0;
}
*/