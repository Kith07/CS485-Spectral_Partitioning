#include <stdio.h>
#include <bits/stdc++.h>
#include <vector>
#include "cuda_eigen.h"
using namespace std;

/*
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

/* Calculate the adjacency matrix from the given input graph/nodes*/
void adj_matrix(string& filename, float* raw, int& n_out){
    /*read file usin ifstream instead*/
    ifstream fin(filename);
    int m;
    fin >> n_out >> m;
    if (!raw)
        return;

    /* initialize the adjacency matrix to all zeros */
    int n = n_out;
    for (int i = 0; i < n*n; ++i)
        raw[i] = 0.0f;

    int u, v;
    /* read each edge and mark it in the symmetric adjacency matrix */
    for (int i = 0; i < m; ++i) {
        fin >> u >> v;
        raw[u + v*n] = 1.0f;
        raw[v + u*n] = 1.0f;
    }
}


/* Calculate centroids for the given points in eigenvector space */
vector<float> centroids(vector<vector<float>>& eigenvecs, vector<int>& nodes, int k) {
    vector<float> c(k, 0.0f);
    for (int idx : nodes)
        for (int j = 0; j < k; ++j)
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
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
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

/* Project points onto axis and sort them */
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
vector<vector<int>> get_partitions(vector<vector<float>>& eigenvecs, int n_parts) {
    int n = eigenvecs.size();
    int k = eigenvecs[0].size();

    /*TODO: create a list of all node indices*/
    vector<int> nodes(n);
    iota(nodes.begin(), nodes.end(), 0);

    /* compute the centroid of all node vectors in eigen space */
    vector<float> c = centroids(eigenvecs, nodes, k);

    /* call the intertial matrix and principal axis funcs*/
    vector<vector<float>> R = inertial_matrix(eigenvecs, nodes, c);
    vector<float> axis = getL(R);

    /* project onto axis and sort using standard c sort*/
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
    float d = 0.85;
    int k_eigen = 10;

    if (argc < 3) {
        printf("Please provide the necessary arguments! Input graph file and number of partitions desired");
        return 1;
    }

    string fname = argv[1];
    int n_parts = atoi(argv[2]);

    int n;
    vector<float> raw;
    vector<vector<float>> eigen_matrix;
    vector<vector<int>> parts;

    clock_t start1, end1;

    /*first read graph size* and allocate memory*/
    adj_matrix(fname, nullptr, n);
    raw.resize(n * n);

    /*call the actual adj matrix func*/
    adj_matrix(fname, raw.data(), n);

    /* derive the eigenvectors and partitions*/
    start1 = clock();
    eigen_computation_cuda(raw.data(), n, k_eigen, eigen_matrix);
    end1 = clock();

    parts = get_partitions(eigen_matrix, n_parts);

    /*print the partitions*/
    for (int p = 0; p < parts.size(); ++p) {
        if (!parts[p].empty()) {
            printf("Partition %d:\n", p);
            for (int node : parts[p])
                printf("%d ", node);
            printf("\n\n");
        }
    }

    /*TODO: move timing to bash script for matching output*/
    /*printf("Time for %d nodes with %d eigenvectors and %d clusters: %.6f seconds\n", n, k_eigen, n_parts, float(end1 - start1) / CLOCKS_PER_SEC);*/

    return 0;
}