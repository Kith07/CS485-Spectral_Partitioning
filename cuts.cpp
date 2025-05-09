// compute_edge_cut.cpp
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0]
             << " <graph_file> <partition_file>\n";
        return 1;
    }
    const string graph_file     = argv[1];
    const string partition_file = argv[2];

    // --- 1) load edges ---
    ifstream gfin(graph_file);
    if (!gfin) { perror("open graph"); return 1; }
    int n, m, idx;
    gfin >> n >> m >> idx;               // header
    vector<pair<int,int>> edges;
    edges.reserve(m);
    int u, v;
    while (gfin >> u >> v) {
        edges.emplace_back(u, v);
    }
    gfin.close();

    // --- 2) load partition mapping ---
    ifstream pfin(partition_file);
    if (!pfin) { perror("open partition"); return 1; }
    vector<int> part_of(n, -1);
    string line;
    int current_part = -1;
    while (getline(pfin, line)) {
        if (line.empty()) continue;
        if (line.rfind("Partition", 0) == 0) {
            // "Partition 2:" → extract 2
            int colon = line.find(':');
            current_part = stoi(line.substr(10, colon-10));
        } else {
            istringstream iss(line);
            while (iss >> u) {
                if (u < 0 || u >= n) {
                    cerr << "Node ID " << u << " out of range\n";
                    return 1;
                }
                part_of[u] = current_part;
            }
        }
    }
    pfin.close();

    // verify we assigned all nodes
    for (int i = 0; i < n; i++) {
        if (part_of[i] < 0) {
            cerr << "Warning: node " << i << " missing in partition file\n";
        }
    }

    // --- 3) compute edge-cut ---
    long long cut = 0;
    for (auto &e : edges) {
        if (part_of[e.first] != part_of[e.second]) {
            cut++;
        }
    }

    cout << "Edge-cut: " << cut
         << "  (out of " << edges.size() << " edges)\n";
    return 0;
}
