#include <climits>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdint>

#ifndef MAX_DEPTH
#define MAX_DEPTH 5
#endif

#ifndef EMPTY
#define EMPTY 0
#endif

class SequentialHash
{
private:
    uint32_t size;
    uint32_t t;
    uint32_t maxIter;
    uint32_t seed;
    std::vector<std::vector<uint32_t>> hashtable;
    std::vector<uint32_t> pos;

    void initTable()
    {
        for (uint32_t j = 0; j < size; j++)
            for (uint32_t i = 0; i < t; i++)
                hashtable[i][j] = EMPTY;
    }

    uint32_t hash(uint32_t function, uint32_t key)
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

    bool rehash(uint32_t &rehashes)
    {
        // If exceeds max rehashing depth, abort.
        if (rehashes > MAX_DEPTH)
            return false;

        rehashes++;

        // Generate new set of hash functions.
        seed = rand(); // Pick a new seed

        // Clear data and map, put values into a buffer.
        std::vector<uint32_t> val_buffer;
        for (uint32_t i = 0; i < t; i++)
            for (uint32_t j = 0; j < size; j++)
                if (hashtable[i][j] != EMPTY)
                    val_buffer.push_back(hashtable[i][j]);

        // Reinitialize the table
        initTable();

        // Re-insert all values.
        for (auto val : val_buffer)
            place(val, 0, 0, rehashes);

        return true;
    }

    bool place(uint32_t key, uint32_t tableID, uint32_t cnt, uint32_t &rehashes)
    {
        // Cycle present, rehash.
        if (cnt == maxIter)
        {
            if (!rehash(rehashes))
                return false;
            place(key, 0, 0, rehashes);
            return true;
        }

        for (uint32_t i = 0; i < t; i++)
        {
            pos[i] = hash(i + 1, key);
            if (hashtable[i][pos[i]] == key)
                return true;
        }

        if (hashtable[tableID][pos[tableID]] != EMPTY)
        {
            uint32_t dis = hashtable[tableID][pos[tableID]];
            hashtable[tableID][pos[tableID]] = key;
            return place(dis, (tableID + 1) % t, cnt + 1, rehashes);
        }
        else
        {
            hashtable[tableID][pos[tableID]] = key;
            return true;
        }
    }

public:
    SequentialHash(uint32_t size, uint32_t t, uint32_t maxIter) : size(size), t(t), maxIter(maxIter), seed(rand()), hashtable(t, std::vector<uint32_t>(size)), pos(t) {}

    void insertKeys(uint32_t keys[], uint32_t n, uint32_t &rehashes)
    {
        initTable();
        for (uint32_t i = 0; i < n; i++)
            if (!place(keys[i], 0, 0, rehashes))
            {
                std::cout << "Failed to insert key " << keys[i] << " after exceeding max depth." << std::endl;
                return;
            }
    }

    bool lookupKey(uint32_t key)
    {
        for (uint32_t i = 0; i < t; i++)
        {
            uint32_t pos = hash(i + 1, key);
            if (hashtable[i][pos] == key)
                return true;
        }
        return false;
    }

    void printTables()
    {
        std::cout << "Final hash tables:" << std::endl;
        for (uint32_t i = 0; i < t; i++, std::cout << std::endl)
            for (uint32_t j = 0; j < size; j++)
                (hashtable[i][j] == EMPTY) ? std::cout << "- " : std::cout << hashtable[i][j] << " ";

        std::cout << std::endl;
    }
};
