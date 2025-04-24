#include <stdio.h>
#include <bits/stdc++.h>

using namespace std;

/* Calculate the adjacency matrix from the given input graph/nodes */
void adj_matrix(FILE *file, float** graph, int n, float d, int m, int idx){
    int src, dest;

    if(file == NULL){
        printf("Please provide a valid input file!");
        return;
    }

    for(int i = 0; i < n ; i++){
        graph[i] = (float*)malloc(n * sizeof(float));
        for(int j = 0; j < n; ++j){
            graph[i][j] = (1 - d)/float(n);
        }
    }

    while(m--){
        src = 0, dest = 0;
        fscanf(file, "%d", &src);
        fscanf(file, "%d", &dest);
        if(idx == 0){
            graph[dest][src] += d * 1.0;
        }
        else{
            graph[dest - 1][src - 1] += d * 1.0;
        }
    }
}

/* Normalize each column of the adjacency matrix so the sum is equal to 1 */
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

/* Calculate the L1 norm between current and previous rank vectors */
float L1(float *v, int n){
    float sum = 0.0;
    for(int i = 0; i < n; i++){
        sum += fabsf(v[i]);
    }
    return sum;
}

/* Intialize and calculate rank vector until convergence */
void rank_vector(float **graph, float *r, int n, int max_iter, float eps){
    float* r_last = (float*) malloc(n * sizeof(float));
    
    for(int i = 0; i < n; i++){
        r[i] = (1/(float)n);
    }

    /* repetitively multiply the rank vector by row until convergence is achieved */
    while(max_iter--){
        for(int i = 0; i < n; i++){
            r_last[i] = r[i];
        }

        for(int i = 0; i < n; i++){
            float sum = 0.0;
            for (int j = 0; j < n; j++){
                sum += r_last[j] * graph[i][j];
            }
            r[i] = sum;
        }

        for(int i = 0; i < n; i++){
            r_last[i] -= r[i];
        }

        /* terminate if L1 norm difference is minimal */
        if(L1(r_last, n) < eps){
            return;
        }

    }
    return;
}

/* Output the highest ranked nodes from the given graph using a priority queue */
void high_rank(float *r, int n, int count){
    priority_queue<pair<float, int>> pq;

    for(int i = 0; i < n; i++){
        pq.push(make_pair(r[i], i));
    }
    
    int rank = 1;
    while(rank <= count){
        printf("Rank %d Node: %d\n", rank, pq.top().second);
        rank++;
        pq.pop();
    }
}

int main(int argc, char** argv){
    clock_t start, end;
    FILE *file;
    int n, m, idx;
    int count = 10, max_iter = 1000;
    float d = 0.85, eps = 0.000001;

    char * filename = argv[1];
    file = fopen(filename, "r");

    fscanf(file, "%d %d %d", &n, &m, &idx);

    float** graph = (float**)malloc(n * sizeof(float*));
    float* r = (float*) malloc(n * sizeof(float));

    adj_matrix(file, graph, n, d, m, idx);

    start = clock();

    norm_adj_matrix(graph, n);
    rank_vector(graph, r, n, max_iter, eps);
    high_rank(r, n, count);

    end = clock();

    printf("Time for %d nodes: %f seconds", n, (float(end - start) / CLOCKS_PER_SEC));
    return 0;
}