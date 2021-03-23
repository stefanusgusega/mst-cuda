#include <iostream>
#include <utility>
#include <algorithm>
#include <sys/time.h>

using namespace std;

const int INF = 1e9 + 7;

bool cmp(pair<int, int> a, pair<int, int> b) {
    if(a.first == b.first) {
        return a.second < b.second;
    } else {
        return a.first < b.first;
    }
}

__global__ 
void reduceMin(int n ,pair<int, int> *min_edge, bool *visited, int* minval, int* idxmin)
{
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int localmin = INF;
    int localidxmin = -1;
    for(int j = index; j < n; j += stride) {
        if(visited[j] == 0 && (localidxmin == -1 || min_edge[j].first < min_edge[localidxmin].first)) {
            localidxmin = j;
            localmin = min_edge[j].first;
        }
    }
 
    atomicMin(minval, localmin);
  
    __syncthreads();
    
    if(*minval == localmin) {
        *idxmin = localidxmin;
    }
}

int main(){
    
    int n;
    
    cin >> n;

    int *adj, *idxmin, *minval;
    pair<int, int> *min_edge, *result;
    bool *visited;

    cudaMallocManaged(&adj, n * n * sizeof(int));
    cudaMallocManaged(&idxmin, sizeof(int));
    cudaMallocManaged(&minval, sizeof(int));
    cudaMallocManaged(&min_edge, n * sizeof(pair<int, int>));
    cudaMallocManaged(&result, n * sizeof(pair<int, int>));
    cudaMallocManaged(&visited, n * sizeof(bool));


    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cin >> adj[i * n + j];
            // akses tidak bisa [][]. harus [], maka diflatten
            if(adj[i * n + j] == -1) adj[i * n + j] = INF;
        }

        visited[i] = 0;
        // first: weight, second: terhubung sama apa
        min_edge[i].first = INF;
        min_edge[i].second = -1;
    }

    int total_weight = 0;
    min_edge[0].first = 0;

    int cur = 0;

    struct timeval stop, start;
    gettimeofday(&start, NULL);
    for(int i = 0; i < n; i++) {
        int blockSize = 256;
        
        int numBlocks = (n + blockSize - 1) / blockSize;

        *idxmin = -1;
        *minval = INF;
        reduceMin<<<numBlocks, blockSize>>>(n, min_edge, visited, minval, idxmin);

        cudaDeviceSynchronize();

        int t = *idxmin;
        visited[t] = 1;
        total_weight += min_edge[t].first;
        if(min_edge[t].second != -1) {
            result[cur].first = min(t, min_edge[t].second);
            result[cur].second = max(t, min_edge[t].second);
            cur++;
        }
        //cout << *idxmin << " this is " << *minval << '\n';
        for(int to = 0; to < n; to++) {
            if(adj[t * n + to] < min_edge[to].first) {
                min_edge[to].first = adj[t * n + to];
                min_edge[to].second = t;
            }
            //cout << min_edge[to].first << " - " << min_edge[to].second << '\n';

        }
    }
    gettimeofday(&stop, NULL);

    sort(result, result + cur, cmp);

    cout << total_weight << '\n';

    for(int i = 0; i < cur; i++) {
        cout << result[i].first << '-' << result[i].second << '\n';
    }

    cout << "Waktu Eksekusi: " << (stop.tv_sec - start.tv_sec) * 1000 + (stop.tv_usec - start.tv_usec) / 1000 << " ms\n";
    cudaFree(adj);
    cudaFree(idxmin);
    cudaFree(minval);
    cudaFree(min_edge);
    cudaFree(result);
    cudaFree(visited);
    
    return 0;
}