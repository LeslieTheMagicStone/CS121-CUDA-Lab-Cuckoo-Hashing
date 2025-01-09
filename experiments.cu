#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include "parallel.cuh"
#include "vanilla.hpp"

using namespace std;
using namespace chrono;

void generateRandomKeys(vector<int> &keys, int n)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, INT_MAX);

    for (int i = 0; i < n; ++i)
    {
        keys.push_back(dis(gen));
    }
}

void experiment1(int t, int maxIter)
{
    for (int exp = 10; exp <= 24; ++exp)
    {
        int n = pow(2, exp);
        int size = pow(2, exp);
        vector<int> keys;
        generateRandomKeys(keys, n);

        auto start = high_resolution_clock::now();
        SequentialHash sh(size, t, maxIter);
        sh.insertKeys(keys.data(), n);
        auto end = high_resolution_clock::now();
        auto durationSeq = duration_cast<microseconds>(end - start).count();

        start = high_resolution_clock::now();
        ParallelHash ph(size, t, maxIter);
        ph.insertKeys(keys.data(), n);
        end = high_resolution_clock::now();
        auto durationPar = duration_cast<microseconds>(end - start).count();

        cout << "Experiment1 t=" << t << " exp=" << exp << " [Sequential] " << durationSeq << " us | [Parallel] " << durationPar << " us" << endl;
    }
}

void experiment2(int t, int maxIter)
{
    for (int exp = 10; exp <= 24; ++exp)
    {
        int n = pow(2, exp);
        int size = pow(2, exp);
        vector<int> keys;
        generateRandomKeys(keys, n);

        SequentialHash sh(size, t, maxIter);
        sh.insertKeys(keys.data(), n);

        vector<int> lookupKeys;
        for (int i = 0; i < n; ++i)
        {
            if (i < 0.9 * n)
                lookupKeys.push_back(keys[i]);
            else
                lookupKeys.push_back(rand());
        }

        auto start = high_resolution_clock::now();
        // Perform lookups in SequentialHash
        auto end = high_resolution_clock::now();
        auto durationSeq = duration_cast<microseconds>(end - start).count();

        ParallelHash ph(size, t, maxIter);
        ph.insertKeys(keys.data(), n);

        start = high_resolution_clock::now();
        // Perform lookups in ParallelHash
        end = high_resolution_clock::now();
        auto durationPar = duration_cast<microseconds>(end - start).count();

        cout << "Experiment2 t=" << t << " exp=" << exp << " [Sequential] " << durationSeq << " us | [Parallel] " << durationPar << " us" << endl;
    }
}

void experiment3(int t, int maxIter)
{
    int n = pow(2, 20);
    vector<int> keys;
    generateRandomKeys(keys, n);

    for (double factor = 1.1; factor <= 2.0; factor += 0.1)
    {
        int size = n * factor;

        auto start = high_resolution_clock::now();
        SequentialHash sh(size, t, maxIter);
        sh.insertKeys(keys.data(), n);
        auto end = high_resolution_clock::now();
        auto durationSeq = duration_cast<microseconds>(end - start).count();

        start = high_resolution_clock::now();
        ParallelHash ph(size, t, maxIter);
        ph.insertKeys(keys.data(), n);
        end = high_resolution_clock::now();
        auto durationPar = duration_cast<microseconds>(end - start).count();

        cout << "Experiment3 t=" << t << " factor=" << factor << " [Sequential] " << durationSeq << " us | [Parallel] " << durationPar << " us" << endl;
    }
}

void experiment4(int t, int n)
{
    int size = n * 1.4;
    vector<int> keys;
    generateRandomKeys(keys, n);

    for (int bound = 1; bound <= 10; ++bound)
    {
        auto start = high_resolution_clock::now();
        SequentialHash sh(size, t, bound);
        sh.insertKeys(keys.data(), n);
        auto end = high_resolution_clock::now();
        auto durationSeq = duration_cast<microseconds>(end - start).count();

        start = high_resolution_clock::now();
        ParallelHash ph(size, t, bound);
        ph.insertKeys(keys.data(), n);
        end = high_resolution_clock::now();
        auto durationPar = duration_cast<microseconds>(end - start).count();

        cout << "Experiment4 t=" << t << " bound=" << bound << " [Sequential] " << durationSeq << " us | [Parallel] " << durationPar << " us" << endl;
    }
}

int main()
{
    int t = 2;
    int maxIter = 4 * log2(10);

    cout << "Experiment 1:" << endl;
    experiment1(t, maxIter);

    cout << "Experiment 2:" << endl;
    experiment2(t, maxIter);

    cout << "Experiment 3:" << endl;
    experiment3(t, maxIter);

    cout << "Experiment 4:" << endl;
    experiment4(t, pow(2, 20));

    return 0;
}
