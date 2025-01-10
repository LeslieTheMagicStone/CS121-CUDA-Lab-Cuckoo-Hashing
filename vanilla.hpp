#include <climits>
#include <cstdio>
#include <vector>

class SequentialHash
{
private:
    int size;
    int t;
    int maxIter;
    std::vector<std::vector<int>> hashtable;
    std::vector<int> pos;

    void initTable()
    {
        for (int j = 0; j < size; j++)
            for (int i = 0; i < t; i++)
                hashtable[i][j] = INT_MIN;
    }

    int hash(int function, int key)
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

    void place(int key, int tableID, int cnt, int &rehashes)
    {
        // Cycle present, rehash.
        if (cnt == maxIter)
        {
            rehashes++;
            return;
        }

        for (int i = 0; i < t; i++)
        {
            pos[i] = hash(i + 1, key);
            if (hashtable[i][pos[i]] == key)
                return;
        }

        if (hashtable[tableID][pos[tableID]] != INT_MIN)
        {
            int dis = hashtable[tableID][pos[tableID]];
            hashtable[tableID][pos[tableID]] = key;
            place(dis, (tableID + 1) % t, cnt + 1, rehashes);
        }
        else
        {
            hashtable[tableID][pos[tableID]] = key;
        }
    }

public:
    SequentialHash(int size, int t, int maxIter) : size(size), t(t), maxIter(maxIter), hashtable(t, std::vector<int>(size)), pos(t) {}

    void insertKeys(int keys[], int n, int &rehashes)
    {
        initTable();
        for (int i = 0, cnt = 0; i < n; i++, cnt = 0)
            place(keys[i], 0, cnt, rehashes);
    }

    void printTables()
    {
        printf("Final hash tables:\n");
        for (int i = 0; i < t; i++, printf("\n"))
            for (int j = 0; j < size; j++)
                (hashtable[i][j] == INT_MIN) ? printf("- ") : printf("%d ", hashtable[i][j]);

        printf("\n");
    }
};
