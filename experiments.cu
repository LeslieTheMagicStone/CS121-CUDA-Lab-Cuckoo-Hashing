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
    uint32_t n = 8;
    uint32_t t = 2;
    uint32_t size = 10;
    uint32_t maxIter = 4 * log2(n);
    vector<uint32_t> keys = {1, 2, 3, 4, 5, 6, 7, 8};

    std::cout << "Running simple demo with n=8, t=2, size=10" << std::endl;

    uint32_t rehashesSeq = 0;
    SequentialHash sh(size, t, maxIter);
    sh.insertKeys(keys.data(), n, rehashesSeq);

    std::cout << "Final table:" << std::endl;
    sh.printTables();

    uint32_t rehashesPar = 0;
    ParallelHash ph(size, t, maxIter);
    ph.insertKeys(keys.data(), n, rehashesPar);

    std::cout << "Final table:" << std::endl;
    ph.printTables();
}

void experiment1(uint32_t t)
{
    for (uint32_t exp = 10; exp <= 24; ++exp)
    {
        for (int iter = 0; iter < 5; iter++)
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

            uint32_t rehashesPar = 0;
            ParallelHash ph(size, t, maxIter);
            start = high_resolution_clock::now();
            ph.insertKeys(keys.data(), n, rehashesPar);
            end = high_resolution_clock::now();
            auto durationPar = duration_cast<microseconds>(end - start).count();

            printf("E1-%d t=%u exp=%u [Sequential] %8ld us, rehashes: %4u | [CUDA] %8ld us, rehashes: %4u\n", iter, t, exp, durationSeq, rehashesSeq, durationPar, rehashesPar);
        }
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

    ParallelHash ph(size, t, maxIter);
    uint32_t rehashesPar = 0;
    ph.insertKeys(keys.data(), n, rehashesPar);

    for (uint32_t i = 0; i <= 10; ++i)
    {
        for (int iter = 0; iter < 5; iter++)
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

            bool* results = new bool[n];
            start = high_resolution_clock::now();
            ph.lookupKeys(S[i].data(), results, n);
            end = high_resolution_clock::now();
            auto durationPar = duration_cast<microseconds>(end - start).count();
            delete [] results;

            printf("E2-%d t=%u i=%-2u [Sequential] %8ld us | [CUDA] %8ld us\n", iter, t, i, durationSeq, durationPar);
        }
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
        for (int iter = 0; iter < 5; iter++)
        {
            uint32_t size = (uint32_t)(i * n);

            SequentialHash sh(size, t, maxIter);
            uint32_t rehashesSeq = 0;
            auto start = high_resolution_clock::now();
            sh.insertKeys(keys.data(), n, rehashesSeq);
            auto end = high_resolution_clock::now();
            auto durationSeq = duration_cast<microseconds>(end - start).count();

            ParallelHash ph(size, t, maxIter);
            uint32_t rehashesPar = 0;
            start = high_resolution_clock::now();
            ph.insertKeys(keys.data(), n, rehashesPar);
            end = high_resolution_clock::now();
            auto durationPar = duration_cast<microseconds>(end - start).count();

            printf("E3-%d t=%u size=%.2lfn [Sequential] %8ld us, rehashes: %4u | [CUDA] %8ld us, rehashes: %4u\n", iter, t, i, durationSeq, rehashesSeq, durationPar, rehashesPar);
        }
    }
}

void experiment4(uint32_t t)
{
    uint32_t n = pow(2, 24);
    uint32_t size = n * 14 / 10;
    vector<uint32_t> keys;
    generateRandomKeys(keys, n);

    int bestAlphaSeq = 0;
    long bestTimeSeq = LONG_MAX;

    int bestAlphaPar = 0;
    long bestTimePar = LONG_MAX;

    for (int i = 2; i <= 10; i++)
    {
        for (int iter = 0; iter < 5; iter++)
        {
            uint32_t maxIter = i * log2(n);

            SequentialHash sh(size, t, maxIter);
            uint32_t rehashesSeq = 0;
            auto start = high_resolution_clock::now();
            sh.insertKeys(keys.data(), n, rehashesSeq);
            auto end = high_resolution_clock::now();
            auto durationSeq = duration_cast<microseconds>(end - start).count();

            if (durationSeq < bestTimeSeq)
            {
                bestTimeSeq = durationSeq;
                bestAlphaSeq = i;
            }
            
            ParallelHash ph(size, t, maxIter);
            uint32_t rehashesPar = 0;
            start = high_resolution_clock::now();
            ph.insertKeys(keys.data(), n, rehashesPar);
            end = high_resolution_clock::now();
            auto durationPar = duration_cast<microseconds>(end - start).count();

            if (durationPar < bestTimePar)
            {
                bestTimePar = durationPar;
                bestAlphaPar = i;
            }

            printf("E4-%d t=%u maxIter=%d logn [Sequential] %8ld us, rehashes: %4u | [CUDA] %8ld us, rehashes: %4u\n", iter, t, i, durationSeq, rehashesSeq, durationPar, rehashesPar);
        }
    }

    printf ("Best alpha for Sequential: %d\n", bestAlphaSeq);
    printf ("Best alpha for CUDA: %d\n", bestAlphaPar);
}

int main()
{
    simpleDemo();

    printf("Experiment 1:\n");
    experiment1(2);
    experiment1(3);

    printf("Experiment 2:\n");
    experiment2(2);
    experiment2(3);

    printf("Experiment3:\n");
    experiment3(2);
    experiment3(3);

    printf("Experiment4:\n");
    experiment4(2);
    experiment4(3);

    return 0;
}
