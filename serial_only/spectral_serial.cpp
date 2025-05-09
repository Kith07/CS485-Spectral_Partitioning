#include <stdio.h>
#include <bits/stdc++.h>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace std;
using namespace Eigen;

/* Calculate the adjacency matrix from the given input graph/nodes
void adj_matrix(FILE *file, float** graph, int n, float d, int m, int idx) {
    int src, dest;
    
    for (int i = 0; i < n; i++) {
        graph[i] = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
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

/* project points onto axis and sort them */
vector<pair<float, int>> project(vector<vector<float>>& eigenvecs, vector<int>& nodes, vector<float>& axis) {
    vector<pair<float, int>> prjs;
    
    for (int i = 0; i < nodes.size(); i++) {
        float dot = 0.0f;
        for (int j = 0; j < axis.size(); j++)
            dot += eigenvecs[nodes[i]][j] * axis[j];
        prjs.emplace_back(dot, nodes[i]);
    }
    
    return prjs;
}

/* spectral parition with k eigenvectors into n_parts clusters */
vector<vector<int>> spectral_partition(float** graph, int n, int k_eigen, int n_parts, MatrixXf& eigen_matrix) {
    MatrixXf A(n, n), D = MatrixXf::Zero(n, n);

    for (int i = 0; i < n; i++) {
        float row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            A(i, j) = graph[i][j];
            row_sum += graph[i][j];
        }
        D(i, i) = row_sum;
    }

    /*MatrixXf L = D - A;*/
    /*TODO: call the norm function here*/
    MatrixXf L = norm_lap(A);

    /* actual eigensolver function from the library*/
    SelfAdjointEigenSolver<MatrixXf> eigensolver(L);
    eigen_matrix = eigensolver.eigenvectors().leftCols(k_eigen);
    /*
    printf("Top 10 eigenvectors computed\n");
    */

    /*REMEMBER: standardaize the sign of the eigenvectors for matching output*/
    for(int j = 0; j < k_eigen; j++) {
        float val_max = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = eigen_matrix(i, j);
            if (fabs(v) > fabs(val_max))
                val_max = v;
        }
        if (val_max < 0.0f)
            eigen_matrix.col(j) *= -1;
    }

    /* feed the eigen matrix into the eigenvecs array*/
    vector<vector<float>> eigenvecs(n, vector<float>(k_eigen));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k_eigen; j++)
            eigenvecs[i][j] = eigen_matrix(i, j);
            
    /*TODO: create a list of all node indices*/
    vector<int> nodes(n);
    iota(nodes.begin(), nodes.end(), 0);
    
    /* call the cetroid func*/
    vector<float> c = centroids(eigenvecs, nodes, k_eigen);
    
    /* call the intertial matrix func and get the principal axis*/
    vector<vector<float>> R = inertial_matrix(eigenvecs, nodes, c);
    vector<float> axis = getL(R);
    
    vector<pair<float, int>> prjs = project(eigenvecs, nodes, axis);
    
    /*TODO: round to 4 decimal places to get identical eigenvectors/values*/
    for (auto &p : prjs) {
        p.first = std::round(p.first * 10000.0f) / 10000.0f;
    }

    /*TODO: standard cpp sort on the projected node values*/
    sort(prjs.begin(), prjs.end());
    /*printf("Projection and sorting on nodes complete.\n");*/
    
    vector<vector<int>> parts(n_parts);
    int per_part = n / n_parts;
    int rem = n % n_parts;
    int idx = 0;
    
    /* split the nodes on the projected axis evenly among n partitions as desired*/
    /* TODO: recursive bisection for optimal clusters*/
    for (int c = 0; c < n_parts; c++) {
        int count;
        if(c < rem)
            count = per_part + 1;
        else
            count = per_part;
        for (int j = 0; j < count && idx < n; j++) {
            parts[c].push_back(prjs[idx++].second);
            /*DEBUG: printf("Node %d to partition %d\n", prjs[idx-1].second, c);*/
        }
    }

    return parts;
}

int main(int argc, char** argv) {
    srand(0);
    clock_t start, end;
    FILE* file;
    int n, m, idx;
    int k_eigen = 10, n_parts = 5;
    float d = 0.85;

    if (argc < 2) {
        printf("Please provide the necessary arguments! Input graph file and number of partitions desired");
        return 1;
    }

    if (argc >= 3)
        n_parts = atoi(argv[2]);

    /* load the adjacency matrix from the input file*/
    string fname = (string) argv[1];
    MatrixXf A = adj_matrix(fname);
    n = A.rows();

    /* format the adj matrix into a 2d array*/
    float** graph = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        graph[i] = (float*)malloc(n * sizeof(float));
        /*printf("graph[%d]\n", i);*/
        for (int j = 0; j < n; j++)
            graph[i][j] = A(i, j);
    }

    MatrixXf eigen_matrix;

    start = clock();
    vector<vector<int>> parts = spectral_partition(graph, n, k_eigen, n_parts, eigen_matrix);
    end = clock();

    for (int i = 0; i < parts.size(); i++) {
        if (!parts[i].empty()) {
            printf("Partition %d:\n", i);
            for (int node : parts[i])
                printf("%d ", node);
            printf("\n\n");
        }
    }

    /*printf("Time for %d nodes with %d eigenvectors and %d clusters: %.6f seconds\n", n, k_eigen, n_parts, float(end - start) / CLOCKS_PER_SEC);*/

    /*TODO: clear up the 2d array memory*/
    for (int i = 0; i < n; i++)
        free(graph[i]);
        
    free(graph);

    return 0;
}