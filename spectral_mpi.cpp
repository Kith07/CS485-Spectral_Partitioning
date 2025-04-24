#include <stdio.h>
#include <bits/stdc++.h>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include <mpi.h>

using namespace std;
using namespace Eigen;

/*
void norm_adj_matrix(float** graph, int n){
    for(int j = 0; j < n; j++){
        float sum = 0.0;
        for(int i = 0; i < n; i++){
            sum += graph[i][j];
        }

        for(int i = 0; i < n; i++){
            if (sum != 0.0){
                graph[i][j] /= sum;
            }
            else{
                graph[i][j] = (1/(float)n);
            }
        }
    }
}
*/

/* Calculate the L1 norm between current and previous rank vectors
float L1(float *v, int n){
    float sum = 0.0;
    for(int i = 0; i < n; i++){
        sum += fabsf(v[i]);
    }
    return sum;
}
*/

/* scrap L1 and normalize the vectors ?*/
void norm(float* x, int n, MPI_Comm comm) {
    float local_sum = 0.0;
    
    for (int i = 0; i < n; i++) 
        local_sum += x[i] * x[i];
    
    float global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, comm);
    
    float norm = sqrt(global_sum);
    for (int i = 0; i < n; i++) 
        x[i] /= norm;
}

MatrixXf mpi_power_iteration_k(MatrixXf& L, int n, int k, int max_iter, float eps, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MatrixXf Q(n, k);
    for (int col = 0; col < k; ++col) {
        VectorXf x = VectorXf::Random(n);
        norm(x.data(), n, comm);
        VectorXf y_local(n);

        for (int iter = 0; iter < max_iter; iter++) {
            y_local.setZero();
            for (int i = rank; i < n; i += size) {
                for (int j = 0; j < n; j++) {
                    y_local[i] += L(i, j) * x(j);
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, y_local.data(), n, MPI_FLOAT, MPI_SUM, comm);

            /* Gram-Schmidt orthogonalization against previous vectors */
            for (int prev = 0; prev < col; ++prev) {
                float dot = x.dot(Q.col(prev));
                x -= dot * Q.col(prev);
            }
            norm(x.data(), n, comm);
        }
        Q.col(col) = x;
    }
    return Q;
}

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

vector<vector<int>> spectral_partition(float** graph, int n, int k, MatrixXf& eigen_matrix, MPI_Comm comm) {
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

    eigen_matrix = mpi_power_iteration_k(L, n, k, 100, 1e-6, comm);

    /* K-means clustering on eigen_matrix rows */
    MatrixXf features = eigen_matrix;
    MatrixXf centroids = MatrixXf::Zero(k, k);
    default_random_engine rng;
    uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < k; i++) centroids.row(i) = features.row(dist(rng));

    vector<int> labels(n);
    for (int iter = 0; iter < 50; iter++) {
        vector<int> counts(k, 0);
        MatrixXf new_centroids = MatrixXf::Zero(k, k);

        for (int i = 0; i < n; i++) {
            int best = 0;
            float best_dist = (features.row(i) - centroids.row(0)).squaredNorm();
            for (int j = 1; j < k; j++) {
                float dist = (features.row(i) - centroids.row(j)).squaredNorm();
                if (dist < best_dist) {
                    best = j;
                    best_dist = dist;
                }
            }
            labels[i] = best;
            new_centroids.row(best) += features.row(i);
            counts[best]++;
        }
        for (int j = 0; j < k; j++)
            if (counts[j] > 0) centroids.row(j) = new_centroids.row(j) / counts[j];
    }

    vector<vector<int>> clusters(k);
    for (int i = 0; i < n; i++) clusters[labels[i]].push_back(i);
    return clusters;
}

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
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    clock_t start, end;
    FILE *file;
    int n, m, idx;
    int count = 10, k = 3, node = -1;
    float d = 0.85;

    if (argc > 2)
        node = atoi(argv[2]);

    file = fopen(argv[1], "r");
    fscanf(file, "%d %d %d", &n, &m, &idx);

    float** graph = (float**)malloc(n * sizeof(float*));
    adj_matrix(file, graph, n, d, m, idx);

    start = clock();

    MatrixXf eigen_matrix;
    spectral_partition(graph, n, k, eigen_matrix, MPI_COMM_WORLD);

    if (node >= 0 && node < n) {
        auto pages = retrieve_nodes(eigen_matrix, node, count);
        if (rank == 0) {
            printf("\nTop %d Nearest Nodes/Pages of Node %d:\n", count, node);
            for (int i = 0; i < pages.size(); ++i) {
                printf("%d. Node %d (Euclidean Distance %.6f)\n", i + 1, pages[i].second, pages[i].first);
            }
        }
    } else if (rank == 0) {
        printf("\nNot a valid Node!\n");
    }

    end = clock();
    if (rank == 0)
        printf("\nTime for %d nodes: %f seconds\n", n, (float(end - start) / CLOCKS_PER_SEC));

    MPI_Finalize();
    return 0;
}