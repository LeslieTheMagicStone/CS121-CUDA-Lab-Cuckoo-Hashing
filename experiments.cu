#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include "parallel.cuh"
#include "vanilla.hpp"

using namespace std;
using namespace chrono;

void generateRandomKeys(vector<uint32_t> &keys, uint32_t n)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<uint32_t> dis(1, UINT32_MAX);

    for (uint32_t i = 0; i < n; ++i)
    {
        keys.push_back(dis(gen));
    }
}

void simpleDemo()
{
    uint32_t n = 5;
    uint32_t t = 2;
    uint32_t size = 10;
    uint32_t maxIter = 4 * log2(n);
    vector<uint32_t> keys = {1, 2, 3, 4, 5};

    std::cout << "Running simple demo with n=5, t=2, size=10" << std::endl;

    uint32_t rehashesSeq = 0;
    SequentialHash sh(size, t, maxIter);
    sh.insertKeys(keys.data(), n, rehashesSeq);

    std::cout << "Final table:" << std::endl;
    sh.printTables();
}

void experiment1(uint32_t t)
{
    for (uint32_t exp = 3; exp <= 24; ++exp)
    {
        uint32_t n = pow(2, exp);
        uint32_t size = pow(2, 25);
        uint32_t maxIter = 4 * log2(n);
        vector<uint32_t> keys;
        generateRandomKeys(keys, n);

        uint32_t rehashesSeq = 0;
        SequentialHash sh(size, t, maxIter);
        auto start = high_resolution_clock::now();
        sh.insertKeys(keys.data(), n, rehashesSeq);
        auto end = high_resolution_clock::now();
        auto durationSeq = duration_cast<microseconds>(end - start).count();

        printf("E1 t=%u exp=%u [Sequential] %8ld us, rehashes: %4u \n", t, exp, durationSeq, rehashesSeq);
    }
}

void experiment2(uint32_t t)
{
    uint32_t size = pow(2, 25);
    uint32_t n = pow(2, 24);
    uint32_t maxIter = 4 * log2(n);
    vector<uint32_t> keys;
    generateRandomKeys(keys, n);
    vector<vector<uint32_t>> S(11);

    SequentialHash sh(size, t, maxIter);
    uint32_t rehashesSeq = 0;
    sh.insertKeys(keys.data(), n, rehashesSeq);

    for (uint32_t i = 0; i <= 10; ++i)
    {
        generateRandomKeys(S[i], n * i / 10);
        // Fill remaining S[i] with keys randomly chosen in keys[]
        for (uint32_t j = S[i].size(); j < n; ++j)
        {
            // Pick a random index from [0, n-1]
            uint32_t idx = rand() % n;
            S[i].push_back(keys[idx]);
        }

        auto start = high_resolution_clock::now();
        for (uint32_t k = 0; k < n; ++k)
            sh.lookupKey(S[i][k]);
        auto end = high_resolution_clock::now();
        auto durationSeq = duration_cast<microseconds>(end - start).count();

        printf("E2 t=%u i=%u [Sequential] %8ld us\n", t, i, durationSeq);
    }
}

void experiment3(uint32_t t)
{
    uint32_t n = pow(2, 24);
    uint32_t maxIter = 4 * log2(n);
    vector<uint32_t> keys;
    generateRandomKeys(keys, n);

    vector<double> alpha = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 1.01, 1.02, 1.05};

    for (double i : alpha)
    {
        uint32_t size = (uint32_t)(i * n);
        SequentialHash sh(size, t, maxIter);
        uint32_t rehashesSeq = 0;
        auto start = high_resolution_clock::now();
        sh.insertKeys(keys.data(), n, rehashesSeq);
        auto end = high_resolution_clock::now();
        auto durationSeq = duration_cast<microseconds>(end - start).count();

        printf("E3 t=%u size=%.2lfn [Sequential] %8ld us, rehashes: %4u\n", t, i, durationSeq, rehashesSeq);
    }
}

void experiment4(uint32_t t)
{
    uint32_t n = pow(2, 24);
    uint32_t size = n * 14 / 10;
    vector<uint32_t> keys;
    generateRandomKeys(keys, n);

    int bestAlpha = 0;
    long bestTime = LONG_MAX;

    for (int i = 2; i <= 10; i++)
    {
        uint32_t maxIter = i * log2(n);
        SequentialHash sh(size, t, maxIter);
        uint32_t rehashesSeq = 0;
        auto start = high_resolution_clock::now();
        sh.insertKeys(keys.data(), n, rehashesSeq);
        auto end = high_resolution_clock::now();
        auto durationSeq = duration_cast<microseconds>(end - start).count();

        if (durationSeq < bestTime)
        {
            bestTime = durationSeq;
            bestAlpha = i;
        }

        printf("E4 t=%u maxIter=%dlogn [Sequential] %8ld us, rehashes: %4u\n", t, i, durationSeq, rehashesSeq);
    }

    printf("Best maxIter=%dlogn\n", bestAlpha);
}

int main()
{
    simpleDemo();

    printf ("Experiment 1:\n");
    experiment1(2);
    experiment1(3);

    printf ("Experiment 2:\n");
    experiment2(2);
    experiment2(3);

    printf("Experiment3:\n");
    experiment3(2);
    experiment3(3);

    printf ("Experiment4:\n");
    experiment4(2);
    experiment4(3);

    return 0;
}
