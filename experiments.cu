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

void experiment1(int t)
{
    for (int exp = 10; exp <= 24; ++exp)
    {
        int n = pow(2, exp);
        int size = n;
        int maxIter = 4 * log2(n);
        vector<int> keys;
        generateRandomKeys(keys, n);

        ParallelHash ph(size, maxIter, t);
        auto start = high_resolution_clock::now();
        ph.insertKeys(keys.data(), n);
        auto end = high_resolution_clock::now();
        duration<double> diff = end - start;
        double opsPerSec = (n / diff.count()) / 1e6;
        cout << "Experiment 1 (Parallel, t=" << t << ", n=2^" << exp << "): " << opsPerSec << " Mops/sec" << endl;

        SequentialHash sh(size, maxIter, t);
        start = high_resolution_clock::now();
        sh.insertKeys(keys.data(), n);
        end = high_resolution_clock::now();
        diff = end - start;
        opsPerSec = (n / diff.count()) / 1e6;
        cout << "Experiment 1 (Sequential, t=" << t << ", n=2^" << exp << "): " << opsPerSec << " Mops/sec" << endl;
    }
}

void experiment2(int t)
{
    for (int exp = 10; exp <= 24; ++exp)
    {
        int n = pow(2, exp);
        int size = n;
        int maxIter = 4 * log2(n);
        vector<int> keys;
        generateRandomKeys(keys, n);

        ParallelHash ph(size, maxIter, t);
        ph.insertKeys(keys.data(), n);

        for (int k = 0; k <= 10; ++k)
        {
            int lookupSize = pow(2, exp);
            vector<int> lookupKeys;
            generateRandomKeys(lookupKeys, lookupSize);

            auto start = high_resolution_clock::now();
            // Perform lookups (not implemented in ParallelHash)
            auto end = high_resolution_clock::now();
            duration<double> diff = end - start;
            double opsPerSec = (lookupSize / diff.count()) / 1e6;
            cout << "Experiment 2 (Parallel, t=" << t << ", n=2^" << exp << ", k=" << k << "): " << opsPerSec << " Mops/sec" << endl;

            SequentialHash sh(size, maxIter, t);
            sh.insertKeys(keys.data(), n);

            start = high_resolution_clock::now();
            // Perform lookups (not implemented in SequentialHash)
            end = high_resolution_clock::now();
            diff = end - start;
            opsPerSec = (lookupSize / diff.count()) / 1e6;
            cout << "Experiment 2 (Sequential, t=" << t << ", n=2^" << exp << ", k=" << k << "): " << opsPerSec << " Mops/sec" << endl;
        }
    }
}

void experiment3(int t)
{
    int exp = 20;
    int n = pow(2, exp);
    vector<int> keys;
    generateRandomKeys(keys, n);

    for (double factor = 1.1; factor <= 2.0; factor += 0.1)
    {
        int size = factor * n;
        int maxIter = 4 * log2(n);

        ParallelHash ph(size, maxIter, t);
        auto start = high_resolution_clock::now();
        ph.insertKeys(keys.data(), n);
        auto end = high_resolution_clock::now();
        duration<double> diff = end - start;
        double opsPerSec = (n / diff.count()) / 1e6;
        cout << "Experiment 3 (Parallel, t=" << t << ", size=" << factor << "n): " << opsPerSec << " Mops/sec" << endl;

        SequentialHash sh(size, maxIter, t);
        start = high_resolution_clock::now();
        sh.insertKeys(keys.data(), n);
        end = high_resolution_clock::now();
        diff = end - start;
        opsPerSec = (n / diff.count()) / 1e6;
        cout << "Experiment 3 (Sequential, t=" << t << ", size=" << factor << "n): " << opsPerSec << " Mops/sec" << endl;
    }

    for (double factor : {1.01, 1.02, 1.05})
    {
        int size = factor * n;
        int maxIter = 4 * log2(n);

        ParallelHash ph(size, maxIter, t);
        auto start = high_resolution_clock::now();
        ph.insertKeys(keys.data(), n);
        auto end = high_resolution_clock::now();
        duration<double> diff = end - start;
        double opsPerSec = (n / diff.count()) / 1e6;
        cout << "Experiment 3 (Parallel, t=" << t << ", size=" << factor << "n): " << opsPerSec << " Mops/sec" << endl;

        SequentialHash sh(size, maxIter, t);
        start = high_resolution_clock::now();
        sh.insertKeys(keys.data(), n);
        end = high_resolution_clock::now();
        diff = end - start;
        opsPerSec = (n / diff.count()) / 1e6;
        cout << "Experiment 3 (Sequential, t=" << t << ", size=" << factor << "n): " << opsPerSec << " Mops/sec" << endl;
    }
}

void experiment4(int t)
{
    int exp = 20;
    int n = pow(2, exp);
    int size = 1.4 * n;
    vector<int> keys;
    generateRandomKeys(keys, n);

    for (int maxIter = 1; maxIter <= 10; ++maxIter)
    {
        ParallelHash ph(size, maxIter, t);
        auto start = high_resolution_clock::now();
        ph.insertKeys(keys.data(), n);
        auto end = high_resolution_clock::now();
        duration<double> diff = end - start;
        double opsPerSec = (n / diff.count()) / 1e6;
        cout << "Experiment 4 (Parallel, t=" << t << ", maxIter=" << maxIter << "): " << opsPerSec << " Mops/sec" << endl;

        SequentialHash sh(size, maxIter, t);
        start = high_resolution_clock::now();
        sh.insertKeys(keys.data(), n);
        end = high_resolution_clock::now();
        diff = end - start;
        opsPerSec = (n / diff.count()) / 1e6;
        cout << "Experiment 4 (Sequential, t=" << t << ", maxIter=" << maxIter << "): " << opsPerSec << " Mops/sec" << endl;
    }
}

int main()
{
    // Simple demo to ensure tables work correctly
    int t = 2;
    int n = 10;
    int size = 8;
    int maxIter = 4 * log2(n);
    vector<int> keys;
    generateRandomKeys(keys, n);

    ParallelHash ph(size, maxIter, t);
    ph.insertKeys(keys.data(), n);
    cout << "Simple demo (Parallel, t=" << t << "): Completed" << endl;

    SequentialHash sh(size, t, maxIter);
    sh.insertKeys(keys.data(), n);
    cout << "Simple demo (Sequential, t=" << t << "): Completed" << endl;

    // Print tables
    cout << "Parallel Hash Table:" << endl;
    ph.printTables();

    cout << "Sequential Hash Table:" << endl;
    sh.printTables();

    for (int t = 2; t <= 3; ++t)
    {
        experiment1(t);
        experiment2(t);
        experiment3(t);
        experiment4(t);
    }

    return 0;
}
