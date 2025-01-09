#include <iostream>

__device__ int device_hash(int key, int size, int type)
{
    if (type == 1)
    {
        return key % size;
    }
    else if (type == 2)
    {
        return (key / size) % size;
    }
    return -1; // Invalid type
}

__global__ void cuckoo_insert(int **tables, int *keys, int n, int size, int maxIter, int t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int key = keys[idx];
        int cnt = 0;
        while (cnt < maxIter)
        {
            for (int type = 1; type <= t; ++type)
            {
                int pos = device_hash(key, size, type);
                int *table = tables[type - 1];
                if (atomicCAS(&table[pos], -1, key) == -1)
                    return;
                int temp = table[pos];
                table[pos] = key;
                key = temp;
            }
            cnt++;
        }
    }
}

class ParallelHash
{
private:
    int *d_keys;
    int **d_tables;
    int size;
    int maxIter;
    int t;

public:
    ParallelHash(int size, int maxIter, int t) : size(size), maxIter(maxIter), t(t)
    {
        for (int i = 0; i < t; ++i)
        {
            cudaMalloc(&d_tables[i], size * sizeof(int));
            cudaMemset(d_tables[i], -1, size * sizeof(int));
        }
    }

    ~ParallelHash()
    {
        for (int i = 0; i< t; ++i)
        {
            cudaFree(d_tables[i]);
        }
    }

    void insertKeys(int keys[], int n)
    {
        cudaMalloc(&d_keys, n * sizeof(int));
        cudaMemcpy(d_keys, keys, n * sizeof(int), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        cuckoo_insert<<<numBlocks, blockSize>>>(d_tables, d_keys, n, size, maxIter, t);

        cudaFree(d_keys);
    }

    void printTables()
    {
        int **h_tables = new int *[t];
        for (int i = 0; i < t; ++i)
        {
            h_tables[i] = new int[size];
            cudaMemcpy(h_tables[i], d_tables[i], size * sizeof(int), cudaMemcpyDeviceToHost);
        }

        for (int i = 0; i < t; ++i)
        {
            std::cout << "Table " << i + 1 << ":" << std::endl;
            for (int j = 0; j < size; ++j)
            {
                if (h_tables[i][j] != -1)
                {
                    std::cout << j << ": " << h_tables[i][j] << std::endl;
                }
            }
        }

        for (int i = 0; i < t; ++i)
        {
            delete[] h_tables[i];
        }
        delete[] h_tables;
    }
};