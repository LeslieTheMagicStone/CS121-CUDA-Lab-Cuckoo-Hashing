#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <vector>
#include <iostream>

#ifndef MAX_DEPTH
#define MAX_DEPTH 5
#endif

#ifndef EMPTY
#define EMPTY 0
#endif

__device__ uint32_t do_hash(uint32_t function, uint32_t key, uint32_t seed, uint32_t size)
{
    switch (function)
    {
    case 1:
        // Jenkins one-at-a-time hash with seed
        {
            uint32_t hash = seed;
            const char *data = reinterpret_cast<const char *>(&key);
            for (size_t i = 0; i < sizeof(key); ++i)
            {
                hash += data[i];
                hash += (hash << 10);
                hash ^= (hash >> 6);
            }
            hash += (hash << 3);
            hash ^= (hash >> 11);
            hash += (hash << 15);
            return hash % size;
        }
    case 2:
        // FNV-1a hash with seed
        {
            uint32_t hash = 2166136261u;
            const char *data = reinterpret_cast<const char *>(&key);
            for (size_t i = 0; i < sizeof(key); ++i)
            {
                hash ^= data[i];
                hash *= 16777619;
            }
            return (hash ^ seed) % size;
        }
    case 3:
        // MurmurHash3 with seed
        {
            uint32_t c1 = 0xcc9e2d51;
            uint32_t c2 = 0x1b873593;
            uint32_t r1 = 15;
            uint32_t r2 = 13;
            uint32_t m = 5;
            uint32_t n = 0xe6546b64;

            uint32_t hash = seed;
            uint32_t k = key;

            k *= c1;
            k = (k << r1) | (k >> (32 - r1));
            k *= c2;

            hash ^= k;
            hash = (hash << r2) | (hash >> (32 - r2));
            hash = hash * m + n;

            hash ^= sizeof(key);
            hash ^= (hash >> 16);
            hash *= 0x85ebca6b;
            hash ^= (hash >> 13);
            hash *= 0xc2b2ae35;
            hash ^= (hash >> 16);

            return hash % size;
        }
    }
    return 0;
}

__global__ void cuckooHashInsert(uint32_t *table, uint32_t *keys, uint32_t n, uint32_t t, uint32_t size, uint32_t seed, uint32_t maxIter, int *rehashFlag)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        uint32_t key = keys[idx];
        for (uint32_t i = 0; i < maxIter; ++i)
        {
            for (uint32_t j = 0; j < t; ++j)
            {
                uint32_t pos = do_hash(j + 1, key, seed, size);
                uint32_t old = atomicExch(&table[j * size + pos], key);
                if (old == EMPTY || old == key)
                    return;
                key = old;
            }
        }
        atomicExch(rehashFlag, 1);
    }
}

__global__ void cuckooHashLookup(uint32_t *table, uint32_t *keys, bool *results, uint32_t n, uint32_t t, uint32_t size, uint32_t seed)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        uint32_t key = keys[idx];
        for (uint32_t j = 0; j < t; ++j)
        {
            uint32_t pos = do_hash(j + 1, key, seed, size);
            if (table[j * size + pos] == key)
            {
                results[idx] = true;
                return;
            }
        }
        results[idx] = false;
    }
}

class ParallelHash
{
private:
    uint32_t size;
    uint32_t t;
    uint32_t seed;
    uint32_t *d_table;
    uint32_t maxIter;
    int *d_rehashFlag;
    uint32_t rehashCount;

    void rehash()
    {
        if (rehashCount >= MAX_DEPTH)
        {
            std::cerr << "[CUDA] Exceeded maximum rehash depth. Abort." << std::endl;
            return;
        }

        seed = rand(); // Pick a new seed
        rehashCount++;

        // Copy current table to host
        uint32_t *h_table = new uint32_t[t * size];
        cudaMemcpy(h_table, d_table, t * size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Clear device table
        cudaMemset(d_table, EMPTY, t * size * sizeof(uint32_t));

        // Re-insert all values
        std::vector<uint32_t> val_buffer;
        for (uint32_t i = 0; i < t * size; ++i)
        {
            if (h_table[i] != EMPTY)
                val_buffer.push_back(h_table[i]);
        }

        uint32_t *d_keys;
        cudaMalloc(&d_keys, val_buffer.size() * sizeof(uint32_t));
        cudaMemcpy(d_keys, val_buffer.data(), val_buffer.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

        uint32_t blockSize = 256;
        uint32_t numBlocks = (val_buffer.size() + blockSize - 1) / blockSize;
        int rehashFlag = 0;
        cudaMemcpy(d_rehashFlag, &rehashFlag, sizeof(int), cudaMemcpyHostToDevice);
        cuckooHashInsert<<<numBlocks, blockSize>>>(d_table, d_keys, val_buffer.size(), t, size, seed, maxIter, d_rehashFlag);

        cudaFree(d_keys);
        delete[] h_table;

        cudaMemcpy(&rehashFlag, d_rehashFlag, sizeof(int), cudaMemcpyDeviceToHost);
        if (rehashFlag)
        {
            rehash();
        }
    }

public:
    ParallelHash(uint32_t size, uint32_t t, uint32_t maxIter) : size(size), t(t), seed(rand()), maxIter(maxIter), rehashCount(0)
    {
        cudaMalloc(&d_table, t * size * sizeof(uint32_t));
        cudaMemset(d_table, EMPTY, t * size * sizeof(uint32_t));
        cudaMalloc(&d_rehashFlag, sizeof(int));
    }

    ~ParallelHash()
    {
        cudaFree(d_table);
        cudaFree(d_rehashFlag);
    }

    void insertKeys(uint32_t *keys, uint32_t n, uint32_t &rehashes)
    {
        uint32_t *d_keys;
        cudaMalloc(&d_keys, n * sizeof(uint32_t));
        cudaMemcpy(d_keys, keys, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

        uint32_t blockSize = 256;
        uint32_t numBlocks = (n + blockSize - 1) / blockSize;
        int rehashFlag = 0;
        cudaMemcpy(d_rehashFlag, &rehashFlag, sizeof(int), cudaMemcpyHostToDevice);
        cuckooHashInsert<<<numBlocks, blockSize>>>(d_table, d_keys, n, t, size, seed, maxIter, d_rehashFlag);

        cudaFree(d_keys);

        cudaMemcpy(&rehashFlag, d_rehashFlag, sizeof(int), cudaMemcpyDeviceToHost);
        if (rehashFlag)
        {
            rehash();
        }
        rehashes = rehashCount;
    }

    void lookupKeys(uint32_t *keys, bool *results, uint32_t n)
    {
        uint32_t *d_keys;
        bool *d_results;
        cudaMalloc(&d_keys, n * sizeof(uint32_t));
        cudaMalloc(&d_results, n * sizeof(bool));
        cudaMemcpy(d_keys, keys, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

        uint32_t blockSize = 256;
        uint32_t numBlocks = (n + blockSize - 1) / blockSize;
        cuckooHashLookup<<<numBlocks, blockSize>>>(d_table, d_keys, d_results, n, t, size, seed);

        cudaMemcpy(results, d_results, n * sizeof(bool), cudaMemcpyDeviceToHost);

        cudaFree(d_keys);
        cudaFree(d_results);
    }

    void printTables()
    {
        uint32_t *h_table = new uint32_t[t * size];
        cudaMemcpy(h_table, d_table, t * size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        std::cout << "Final hash tables:" << std::endl;
        for (uint32_t i = 0; i < t; i++, std::cout << std::endl)
            for (uint32_t j = 0; j < size; j++)
                (h_table[i * size + j] == EMPTY) ? std::cout << "- " : std::cout << h_table[i * size + j] << " ";

        std::cout << std::endl;

        delete[] h_table;
    }
};
