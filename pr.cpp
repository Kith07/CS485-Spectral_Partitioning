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
        if (idx == 0) graph[dest][src] += d * 1.0;
        else graph[dest - 1][src - 1] += d * 1.0;
    }
}

/* Normalize each column of the adjacency matrix so the sum is equal to 1 */
void norm_adj_matrix(float** graph, int n) {
    for (int j = 0; j < n; j++) {
        float sum = 0.0;
        for (int i = 0; i < n; i++) sum += graph[i][j];
        for (int i = 0; i < n; i++) graph[i][j] = (sum != 0.0) ? graph[i][j] / sum : (1 / (float)n);
    }
}

/* Calculate the L1 norm between current and previous rank vectors */
float L1(float *v, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) sum += fabsf(v[i]);
    return sum;
}

/* Intialize and calculate rank vector until convergence */
void rank_vector(float **graph, float *r, int n, int max_iter, float eps) {
    float* r_last = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) r[i] = (1 / (float)n);
    while (max_iter--) {
        memcpy(r_last, r, n * sizeof(float));
        for (int i = 0; i < n; i++) {
            float sum = 0.0;
            for (int j = 0; j < n; j++) sum += r_last[j] * graph[i][j];
            r[i] = sum;
        }
        for (int i = 0; i < n; i++) r_last[i] -= r[i];
        if (L1(r_last, n) < eps) return;
    }
}

/* Currently do a binary split with spectral partitioning for input graphs based on +/- */
pair<vector<int>, vector<int>> spectral_partition(float** graph, int n) {
    MatrixXf A(n, n), D = MatrixXf::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        float row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            A(i, j) = graph[i][j];
            row_sum += graph[i][j];
        }
        D(i, i) = row_sum;
    }
    MatrixXf L = D - A;
    SelfAdjointEigenSolver<MatrixXf> eigensolver(L);
    VectorXf fiedler = eigensolver.eigenvectors().col(1);
    vector<int> partA, partB;
    for (int i = 0; i < n; ++i) {
        (fiedler[i] < 0) ? partA.push_back(i) : partB.push_back(i);
    }
    return {partA, partB};
}

/* rank nodes locally within partitioned sub-graphs */
float** partitioned_graph(float** graph, const vector<int>& nodes) {
    int k = nodes.size();
    float** subgraph = (float**)malloc(k * sizeof(float*));
    for (int i = 0; i < k; ++i) {
        subgraph[i] = (float*)malloc(k * sizeof(float));
        for (int j = 0; j < k; ++j) {
            subgraph[i][j] = graph[nodes[i]][nodes[j]];
        }
    }
    return subgraph;
}

/* do a global normalization for outputting the highest valued nodes */
void combined_ranking(float** graph, const vector<int>& node_ids, int max_iter, float eps, vector<pair<float, int>>& all_scores) {
    int k = node_ids.size();
    float* r = (float*)malloc(k * sizeof(float));
    norm_adj_matrix(graph, k);
    rank_vector(graph, r, k, max_iter, eps);
    for (int i = 0; i < k; i++) {
        all_scores.push_back({r[i], node_ids[i]});
    }
}

int main(int argc, char** argv) {
    clock_t start, end;
    FILE *file;
    int n, m, idx;
    int count = 10, max_iter = 1000;
    float d = 0.85, eps = 0.000001;

    file = fopen(argv[1], "r");
    fscanf(file, "%d %d %d", &n, &m, &idx);

    float** graph = (float**)malloc(n * sizeof(float*));
    adj_matrix(file, graph, n, d, m, idx);

    start = clock();

    auto [clusterA, clusterB] = spectral_partition(graph, n);
    float** subgraphA = partitioned_graph(graph, clusterA);
    float** subgraphB = partitioned_graph(graph, clusterB);

    vector<pair<float, int>> all_scores;
    combined_ranking(subgraphA, clusterA, max_iter, eps, all_scores);
    combined_ranking(subgraphB, clusterB, max_iter, eps, all_scores);

    float total = 0.0;
    for (auto& p : all_scores) total += p.first;
    for (auto& p : all_scores) p.first /= total;

    sort(all_scores.begin(), all_scores.end(), greater<pair<float, int>>());
    for (int i = 0; i < min(count, (int)all_scores.size()); i++) {
        printf("Rank %d Node: %d with score %.6f\n", i + 1, all_scores[i].second, all_scores[i].first);
    }

    end = clock();
    printf("\nTime for %d nodes: %f seconds\n", n, (float(end - start) / CLOCKS_PER_SEC));
    return 0;
}