#include <mpi.h>
#include <stdio.h>
#include <set>
#include <unordered_set>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace std;
using namespace Eigen;

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
        if (fscanf(file, "%d%d", &src, &dest) != 2) {
            fprintf(stderr, "Error reading edge (src, dest) from file.\n");
            exit(EXIT_FAILURE);
        }

        if (idx == 0)
            graph[dest][src] += d * 1.0;
        else 
            graph[dest - 1][src - 1] += d * 1.0;
    }
}
*/
/*
void normalize_laplacian(float** graph, int n, Eigen::MatrixXf& A, Eigen::MatrixXf& D) {
    // Calculate degree matrix
    vector<float> degree(n, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            degree[i] += graph[i][j];
        }
        D(i, i) = degree[i];
    }
    
    // Build normalized adjacency matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (degree[i] > 0 && degree[j] > 0)
                A(i, j) = graph[i][j] / sqrt(degree[i] * degree[j]);
            else 
                A(i, j) = 0.0f;
        }
    }
}
*/

MatrixXf adj_matrix(string& filename) {
    /*read file usin ifstream instead*/
    ifstream fin(filename);
    int n, m;
    fin >> n >> m;
    MatrixXf A = MatrixXf::Zero(n, n);

    int u, v;
    for (int i = 0; i < m; i++) {
        fin >> u >> v;
        assert(u >= 0 && u < n && v >= 0 && v < n);
        A(u,v) = A(v,u) = 1.0f;
    }

    return A;
}

/* calculate the normalized laplacian instead*/
MatrixXf norm_lap(MatrixXf& A) {
    int n = A.rows();
    VectorXf d = A.rowwise().sum();
    VectorXf inv_sqrt = d.array().inverse().sqrt();

    for (int i = 0; i < n; i++)
        if (!isfinite(inv_sqrt(i)))
            inv_sqrt(i) = 0.0f;

    MatrixXf Dinv = inv_sqrt.asDiagonal();

    /*
    MatrixXf::L = d - A;
    return L;
    */
    return MatrixXf::Identity(n, n) - Dinv * A * Dinv;
}

void eigen_computation(MatrixXf& L, int k, MatrixXf& eigen_matrix){
    SelfAdjointEigenSolver<MatrixXf> solver(L);
    assert(solver.info() == Success);

    int k_true = min(k, (int)solver.eigenvectors().cols());
    eigen_matrix = solver.eigenvectors().leftCols(k_true);

    for (int j = 0; j < k_true; j++) {
        float max_val = 0.0f;
        for (int i = 0; i < L.rows(); i++) {
            float v = eigen_matrix(i,j);
            if (fabs(v) > fabs(max_val))
                max_val = v;
        }
        if (max_val < 0.0f)
            eigen_matrix.col(j) *= -1;
    }
}

/* Calculate centroids for the given points in eigenvector space */
vector<float> centroids(vector<vector<float>>& eigenvecs, vector<int>& nodes, int k) {
    vector<float> c(k, 0.0f);
    for (int idx : nodes)
        for (int j = 0; j < k; j++)
            c[j] += eigenvecs[idx][j];

    for (float& val : c)
        val /= nodes.size();

    return c;
}

/* Compute the inertial matrix R based on eigenvectors */
vector<vector<float>> inertial_matrix(vector<vector<float>>& eigenvecs, vector<int>& nodes, vector<float>& c) {
    int k = c.size();
    
    vector<vector<float>> R(k, vector<float>(k, 0.0f));
    for (int idx : nodes) {
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                /* compute R from the eigen vectros from the laplacian*/
                R[i][j] += (eigenvecs[idx][i] - c[i]) * (eigenvecs[idx][j] - c[j]);
    }
    
    return R;
}

/* find principal axis (largest eigenvector of R) using power iteration */
vector<float> getL(vector<vector<float>>& R) {
    int k = R.size();
    vector<float> v(k, 1.0f);
    
    /* pwer iteration to find leading eigenvector*/
    int it = 0;
    while(it < 100) {
        vector<float> v_max(k, 0.0f);
        
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                v_max[i] += R[i][j] * v[j];
        
        /* normalize*/
        float norm = sqrt(inner_product(v_max.begin(), v_max.end(), v_max.begin(), 0.0f));
        for (int i = 0; i < k; i++)
            v[i] = v_max[i] / norm;

        it++;
    }

    return v;
}

/* project all of the nodes onto the principal axis L
vector<float> project(vector<vector<float>>& eigen_matrix, vector<int>& nodes, vector<float>& dir) {
    vector<float> proj_nodes(nodes.size());
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        float dot = 0.0f;
        for (size_t j = 0; j < dir.size(); ++j)
            dot += eigen_matrix[nodes[i]][j] * dir[j];
        proj_nodes[i] = dot;
    }

    return proj_nodes;
}
*/

/* TODO: parallel projection of node eigenvectors onto principal axis */
vector<pair<float, int>> project_parallel(vector<vector<float>>& eigen_matrix, vector<int>& nodes, vector<float>& dir, int rank, int size, MPI_Comm comm) {
    int n = nodes.size();
    int k = dir.size();
    
    /* split work and assign portion of work to each rank/process */
    int per_rank = n / size;
    int rem = n % size;
    int start = rank * per_rank + min(rank, rem);
    int count = per_rank + (rank < rem ? 1 : 0);   /* num nodes this rank handles */

    vector<float> l_dot(count);
    vector<int> l_nodes(count);

    /* project assigned subset of nodes onto the principal axis by each process*/
    for (int i = 0; i < count; i++) {
        int node_idx = start + i;
        if (node_idx < n) {
            float dot = 0.0f;
            for (int j = 0; j < k; j++) {
                dot += eigen_matrix[nodes[node_idx]][j] * dir[j];
            }
            l_dot[i] = dot;
            l_nodes[i] = nodes[node_idx];
        }
    }

    /* define a custom MPI struct pair */
    MPI_Datatype type;
    int blocks[2] = {1, 1};
    MPI_Aint disp[2] = {0, sizeof(float)};
    MPI_Datatype types[2] = {MPI_FLOAT, MPI_INT};
    MPI_Type_create_struct(2, blocks, disp, types, &type);
    MPI_Type_commit(&type);

    /*REMEMBER: share local sizes with all other processes*/
    vector<int> recvs(size);
    MPI_Allgather(&count, 1, MPI_INT, recvs.data(), 1, MPI_INT, comm);

    /* calculate displacements for Allgatherv*/
    vector<int> displs(size, 0);
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + recvs[i-1];
    }

    int total_count = 0;
    for (int i = 0; i < size; i++) {
        total_count += recvs[i];
    }

    vector<pair<float, int>> l_pairs(count);
    for (int i = 0; i < count; i++) {
        l_pairs[i] = make_pair(l_dot[i], l_nodes[i]);
    }

    /*TODO: gather all projected nodes from every process to the master rnak */
    vector<pair<float, int>> pairs(total_count);
    MPI_Allgatherv(l_pairs.data(), count, type, pairs.data(), recvs.data(), displs.data(), type, comm);

    MPI_Type_free(&type);

    /* printf("Rank %d: projected %d nodes (total = %d)\n", rank, count, total_count);*/
    return pairs;
}

/* multiple processes can attach per cluster. need not be the smae
int assign_clusters(int s_id, int num_procs, int s) {
    return min(s_id * num_procs / s, num_procs - 1);
}
*/

/* continue to bisect the graph on the prinicpal axis recursively into halfs until user cluster requirement is met
void recursive_bisection(vector<vector<float>>& eigen_matrix, vector<int> nodes, int depth, int max_depth, int s, int s_id, int g_size, int g_rank, MPI_Comm comm, vector<vector<int>>& clusters) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* base case: stop recursion when max clusters met or you run out of nodes and assign ranks to final clusters
    if (depth == max_depth || nodes.size() <= 1) {
        if (g_rank == assign_clusters(s_id, g_size, s)) {
            clusters[s_id] = nodes;
        }
        return;
    }

    int k = eigen_matrix[0].size();

    vector<float> c = centroids(eigen_matrix, nodes, k);
    vector<vector<float>> R = intertial_matrix(eigen_matrix, nodes, c);

    vector<float> dir = getL(R);
    vector<float> proj_nodes = project(eigen_matrix, nodes, dir);

    vector<pair<float, int>> proj_idx;
    for (size_t i = 0; i < nodes.size(); ++i)
        proj_idx.emplace_back(proj_nodes[i], nodes[i]);

    recursive_bisection(eigen_matrix, ids, depth + 1, max_depth, s, snew_id, g_size, g_rank, subcomm, clusters);
    MPI_Comm_free(&subcomm);
}
*/

/* perform spectral partitioning with k even splits on master process only*/
void get_partitions(vector<vector<float>>& eigen_matrix, vector<int>& node_ids, int n_parts, int rank, int size, MPI_Comm comm, vector<vector<int>>& parts) {
    int k = eigen_matrix[0].size();
    /*TODO: splitting nodes evenly by rank/process */
    int n = node_ids.size();
    int per_rank = n / size;
    int remain = n % size;
    int start = rank * per_rank + min(rank, remain);
    int count = (rank < remain ? per_rank + 1 : per_rank);

    vector<int> local_node_ids(node_ids.begin() + start, node_ids.begin() + start + count);

    /* compute centroid of local nodes */
    vector<float> local_sum(k, 0.0f);
    for (int idx : local_node_ids)
        for (int j = 0; j < k; j++)
            local_sum[j] += eigen_matrix[idx][j];

    vector<float> global_sum(k, 0.0f);
    MPI_Allreduce(local_sum.data(), global_sum.data(), k, MPI_FLOAT, MPI_SUM, comm);

    vector<float> centroid(k);
    for (int j = 0; j < k; j++)
        centroid[j] = global_sum[j] / n;

    /* compute local inertial matrix around global centroid */
    vector<vector<float>> R = inertial_matrix(eigen_matrix, local_node_ids, centroid);

    /* reduce inertial matrices to get global R */
    vector<float> local_R_flat(k * k), global_R_flat(k * k);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            local_R_flat[i*k + j] = R[i][j];

    MPI_Allreduce(local_R_flat.data(), global_R_flat.data(), k*k, MPI_FLOAT, MPI_SUM, comm);

    vector<vector<float>> global_R(k, vector<float>(k));
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            global_R[i][j] = global_R_flat[i*k + j];

    /* compute principal axis on rank 0 and broadcast */
    vector<float> direction(k);
    if (rank == 0) {
        direction = getL(global_R);
    }
    MPI_Bcast(direction.data(), k, MPI_FLOAT, 0, comm);

    /* project each local node onto the shared direction */
    vector<pair<float, int>> proj_idx;
    proj_idx.reserve(count);
    for (int id : local_node_ids) {
        float dot = 0.0f;
        for (int j = 0; j < k; j++)
            dot += direction[j] * eigen_matrix[id][j];

        /*TODO: round to 4 decimal places to get identical eigenvectors/values*/
        float p = round(dot * 10000.0f) / 10000.0f;
        proj_idx.emplace_back(p, id);
        /*proj_idx.emplace_back(dot, id);*/
    }
    int local_count = proj_idx.size();

    /* gather sizes from all ranks */
    vector<int> recvs(size), displs(size);
    MPI_Gather(&local_count, 1, MPI_INT, recvs.data(), 1, MPI_INT, 0, comm);

    int total = 0;
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recvs[i-1];
        }
        total = accumulate(recvs.begin(), recvs.end(), 0);
    }

    /* define MPI struct for pair<float,int> */
    MPI_Datatype float_int_type;
    MPI_Type_contiguous(2, MPI_FLOAT, &float_int_type);
    MPI_Type_commit(&float_int_type);

    /* gather projections to master process */
    vector<pair<float,int>> g_proj;
    if (rank == 0) g_proj.resize(total);

    MPI_Gatherv(proj_idx.data(), local_count, float_int_type, g_proj.data(), recvs.data(), displs.data(), float_int_type, 0, comm);

    /* globally sort the projected nodes with tie-breaking on rank 0 */
    if (rank == 0) {
        /*
        printf("DEBUG: projections before sort:\n");
        for (auto &p : g_proj) {
          if (p.second == 8 || p.second == 58) {
            printf("  node %2d → proj = %.8f\n", p.second, p.first);
          }
        }
        */
       /*TODO: swap out with radix sort*/
       stable_sort(g_proj.begin(), g_proj.end());

        /*assign nodes evenly to partitions*/
        parts.clear();
        parts.resize(n_parts);
        int per_part = total / n_parts;
        int rem_p = total % n_parts;
        int idx = 0;
        for (int c = 0; c < n_parts; c++) {
            int cnt = (c < rem_p ? per_part + 1 : per_part);
            for (int j = 0; j < cnt && idx < total; j++) {
                parts[c].push_back(g_proj[idx++].second);
            }
        }
    }

    MPI_Type_free(&float_int_type);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int g_rank, g_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    if (argc < 3) {
        if (g_rank == 0)
            printf("Usage: %s <fname> <n_parts> [k_eigen]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    string fname = argv[1];
    int n_parts = atoi(argv[2]);
    int k_eigen = 10;

    int n;
    vector<vector<float>> eigen_matrix;
    clock_t start1, end1, start2, end2;

    if (g_rank == 0) {
        MatrixXf A = adj_matrix(fname);

        n = A.rows();
        MatrixXf L = norm_lap(A);

        MatrixXf eigenvecs;

        start1 = clock();
        eigen_computation(L, k_eigen, eigenvecs);
        end1 = clock();

        /* TODO: cnvert to vector-of-vector format for mpi broadcast*/
        eigen_matrix.resize(n, vector<float>(k_eigen));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < k_eigen; ++j)
                eigen_matrix[i][j] = eigenvecs(i, j);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (g_rank != 0)
        eigen_matrix.resize(n, vector<float>(k_eigen));

    for (int i = 0; i < n; ++i)
        /*TODO: broadcast the eigenmatrix across all processes */
        MPI_Bcast(eigen_matrix[i].data(), k_eigen, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /*TODO: create a list of all node indices*/
    vector<int> nodes(n);
    iota(nodes.begin(), nodes.end(), 0);

    vector<vector<int>> parts(n_parts);

    start2 = clock();
    get_partitions(eigen_matrix, nodes, n_parts, g_rank, g_size, MPI_COMM_WORLD, parts);
    end2 = clock();

    MPI_Barrier(MPI_COMM_WORLD);

    if (g_rank == 0) {
        for (int c = 0; c < n_parts; ++c) {
            if (!parts[c].empty()) {
                printf("Partition %d:\n", c);
                for (int id : parts[c])
                    printf("%d ", id);
                printf("\n\n");
            }
        }
        /*
        printf("Time for Calculating %d Eigenvectors using Eigen (CPU): %.6f seconds\n", k_eigen, float(end1 - start1) / CLOCKS_PER_SEC);
        printf("Time for Parallel Partitioning %d nodes: %.6f seconds\n", n, float(end2 - start2) / CLOCKS_PER_SEC);
        */
    }

    MPI_Finalize();
    return 0;
}