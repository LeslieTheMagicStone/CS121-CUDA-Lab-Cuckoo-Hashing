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

int main()
{
    int t = 2;
    int n = 10;
    int size = 8;
    int maxIter = 4 * log2(n);
    vector<int> keys;
    generateRandomKeys(keys, n);

    cout << "Sequential Hash Table:" << endl;
    SequentialHash sh(size, t, maxIter);
    sh.insertKeys(keys.data(), n);
    sh.printTables();

    cout << "Parallel Hash Table:" << endl;
    ParallelHash ph(size, t, maxIter);
    ph.insertKeys(keys.data(), n);
    ph.printTables();

    return 0;
}
