#include <stdio.h>
#include <bits/stdc++.h>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace std;
using namespace Eigen;

/* Calculate the adjacency matrix from the given input graph/nodes */
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

/* spectral parition with k eigenvecotrs for input graphs/nodes */
vector<vector<int>> spectral_partition(float** graph, int n, int k, MatrixXf& eigen_matrix) {
    MatrixXf A(n, n), D = MatrixXf::Zero(n, n);
    /* compute D the diagonal matrix for the graph*/
    for (int i = 0; i < n; ++i) {
        float row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            A(i, j) = graph[i][j];
            row_sum += graph[i][j];
        }
        D(i, i) = row_sum;
    }
    
    /*derive the laplacian matrix by subtracting the adjacency matrix from the diagonal matrix*/
    MatrixXf L = D - A;
    SelfAdjointEigenSolver<MatrixXf> eigensolver(L);
    /*IN case eigenvector input greater than number of nodes in the graph
    REMINDER: trasnfer this functionality into the main function*/
    int k_true = min(k, (int)eigensolver.eigenvectors().cols());
    /* get the k eigenvectors */
    eigen_matrix = eigensolver.eigenvectors().leftCols(k_true);

    vector<VectorXf> features(n);
    for (int i = 0; i < n; ++i) 
        features[i] = eigen_matrix.row(i);

    /* k-means to cluster the nodes/pages in the high dimensional eigen space based on its spectral coordiantes */
    vector<int> labels(n);
    vector<VectorXf> centroids(k_true, VectorXf::Zero(k_true));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n - 1);
    
    /* initialize centroids*/
    for (int i = 0; i < k_true; ++i)
        centroids[i] = features[dis(gen)];

    /* iterative k-means for clustering nodes/pages*/
    int iter = 0;
    while(iter < 100) {
        vector<VectorXf> new_centroids(k_true, VectorXf::Zero(k_true));
        vector<int> counts(k_true, 0);

        for (int i = 0; i < n; ++i) {
            float close_dist = numeric_limits<float>::max();
            int cluster = 0;
            
            for (int j = 0; j < k_true; ++j) {
                float dist = (features[i] - centroids[j]).squaredNorm();
                if (dist < close_dist) {
                    close_dist = dist;
                    cluster = j;
                }
            }
            
            labels[i] = cluster;
            new_centroids[cluster] += features[i];
            counts[cluster]++;
        }
        for (int j = 0; j < k_true; ++j)
            if (counts[j] > 0) 
                centroids[j] = new_centroids[j] / counts[j];

        iter++;
    }

    /* vector containing k clusters*/
    vector<vector<int>> partitions(k_true);
    for (int i = 0; i < n; ++i) 
        partitions[labels[i]].push_back(i);
    return partitions;
}

/* retireive the closest nodes/pages to a given input node based on euclidean distance in the eigen space */
vector<pair<float, int>> retrieve_nodes(const MatrixXf& U, int node, int k) {
    VectorXf query = U.row(node);
    vector<pair<float, int>> dists;
    
    for (int i = 0; i < U.rows(); i++) {
        if (i == node) 
            continue;
        float dist = (U.row(i) - query.transpose()).norm();
        dists.push_back({dist, i});
    }
    
    sort(dists.begin(), dists.end());
    
    return vector<pair<float, int>>(dists.begin(), dists.begin() + k);
}

int main(int argc, char** argv) {
    clock_t start, end;
    FILE *file;
    int n, m, idx;
    int count = 10, k = 10, node = -1;
    float d = 0.85;

    if (argc > 2)
        node = atoi(argv[2]);

    file = fopen(argv[1], "r");
    fscanf(file, "%d %d %d", &n, &m, &idx);

    float** graph = (float**)malloc(n * sizeof(float*));
    adj_matrix(file, graph, n, d, m, idx);

    start = clock();

    MatrixXf eigen_matrix;
    spectral_partition(graph, n, k, eigen_matrix);

    if (node >= 0 && node < n) {
        auto pages = retrieve_nodes(eigen_matrix, node, count);
        printf("\nTop %d Nearest Nodes/Pages of Node %d:\n", count, node);
        
        for (int i = 0; i < pages.size(); ++i) {
            printf("%d. Node %d (Euclidean Distance %.6f)\n", i + 1, pages[i].second, pages[i].first);
        }
    }
    else {
        printf("\nNot a valid Node!\n");
    }

    end = clock();
    
    /* divide to convert into secondss*/
    printf("\nTime for %d nodes: %f seconds\n", n, (float(end - start) / CLOCKS_PER_SEC));
    return 0;
}