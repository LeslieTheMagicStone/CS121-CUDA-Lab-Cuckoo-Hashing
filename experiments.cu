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
        auto start = high_resolution_clock::now();
        SequentialHash sh(size, t, maxIter);
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

int main()
{
    simpleDemo();

    cout << "Experiment 1:" << endl;
    experiment1(2);
    experiment1(3);

    cout << "Experiment 2:" << endl;
    experiment2(2);
    experiment2(3);

    return 0;
}
