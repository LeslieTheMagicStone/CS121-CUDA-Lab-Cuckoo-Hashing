#include <climits>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__device__ int device_hash(int function, int key, int size)
{
    switch (function)
    {
    case 1:
        return key % size;
    case 2:
        return (key / size) % size;
    }
    return 0;
}

__global__ void initTable(int *hashtable, int size, int t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * t)
    {
        hashtable[idx] = INT_MIN;
    }
}

__global__ void place(int *hashtable, int *keys, int n, int size, int t, int maxIter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int key = keys[idx];
        int cnt = 0;
        int tableID = 0;
        while (cnt < maxIter)
        {
            int pos = device_hash(tableID + 1, key, size);
            int old = atomicExch(&hashtable[tableID * size + pos], key);
            if (old == INT_MIN || old == key)
                return;
            key = old;
            tableID = (tableID + 1) % t;
            cnt++;
        }
        printf("%d unpositioned\n", key);
        printf("Cycle present. REHASH.\n");
    }
}

class ParallelHash
{
private:
    int size;
    int t;
    int maxIter;
    int *d_hashtable;
    int *d_pos;


public:
    ParallelHash(int size, int t, int maxIter) : size(size), t(t), maxIter(maxIter)
    {
        cudaMalloc(&d_hashtable, t * size * sizeof(int));
        cudaMalloc(&d_pos, t * sizeof(int));
    }

    ~ParallelHash()
    {
        cudaFree(d_hashtable);
        cudaFree(d_pos);
    }

    void insertKeys(int keys[], int n)
    {
        int *d_keys;
        cudaMalloc(&d_keys, n * sizeof(int));
        cudaMemcpy(d_keys, keys, n * sizeof(int), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (size * t + blockSize - 1) / blockSize;
        initTable<<<numBlocks, blockSize>>>(d_hashtable, size, t);

        numBlocks = (n + blockSize - 1) / blockSize;
        place<<<numBlocks, blockSize>>>(d_hashtable, d_keys, n, size, t, maxIter);

        cudaFree(d_keys);
    }

    void printTables()
    {
        int *h_hashtable = new int[t * size];
        cudaMemcpy(h_hashtable, d_hashtable, t * size * sizeof(int), cudaMemcpyDeviceToHost);

        printf("Final hash tables:\n");
        for (int i = 0; i < t; i++, printf("\n"))
            for (int j = 0; j < size; j++)
                (h_hashtable[i * size + j] == INT_MIN) ? printf("- ") : printf("%d ", h_hashtable[i * size + j]);

        printf("\n");
        delete[] h_hashtable;
    }
};
