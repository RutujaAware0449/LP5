#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
using namespace std;

class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int vertices) {
        V = vertices;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected
    }

    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "\nParallel BFS: ";

        while (!q.empty()) {
            int size = q.size();

            #pragma omp parallel for
            for (int i = 0; i < size; i++) {
                int node;

                // Only one thread at a time accesses the queue
                #pragma omp critical
                {
                    if (!q.empty()) {
                        node = q.front();
                        q.pop();
                        cout << node << " (Thread " << omp_get_thread_num() << ") ";
                    }
                }

                // Process neighbors
                for (int j = 0; j < adj[node].size(); j++) {
                    int neighbor = adj[node][j];

                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }

        cout << endl;
    }

    void parallelDFSUtil(int node, vector<bool>& visited) {
        #pragma omp critical
        {
            cout << node << " (Thread " << omp_get_thread_num() << ") ";
        }

        visited[node] = true;

        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); i++) {
            int neighbor = adj[node][i];
            if (!visited[neighbor]) {
                #pragma omp task firstprivate(neighbor)
                {
                    parallelDFSUtil(neighbor, visited);
                }
            }
        }

        #pragma omp taskwait
    }

    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        cout << "\nParallel DFS: ";
        #pragma omp parallel
        {
            #pragma omp single
            {
                cout << "\nTotal threads in DFS: " << omp_get_num_threads() << endl;
                parallelDFSUtil(start, visited);
            }
        }
        cout << endl;
    }
};

int main() {
    Graph g(8);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);
    g.addEdge(6, 7);

    // Measure BFS time
    double bfs_start = omp_get_wtime();
    g.parallelBFS(0);
    double bfs_end = omp_get_wtime();
    cout << "\nBFS Execution Time: " << (bfs_end - bfs_start) << " seconds\n";

    // Measure DFS time
    double dfs_start = omp_get_wtime();
    g.parallelDFS(0);
    double dfs_end = omp_get_wtime();
    cout << "\nDFS Execution Time: " << (dfs_end - dfs_start) << " seconds\n";

    return 0;
}

