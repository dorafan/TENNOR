#pragma once
#include "Bucket.h"
#include <random>

#include "obl.h"

class LSH {
private:

	int _K;
	int _L;
	int _RangePow;
	int *rand1;

public:
    Bucket ** _bucket;
	LSH(int K, int L, int RangePow);
	void clear();
	int * hashesToIndex(int * hashes);
	int retrieve(int table, int indices, int bucket);
	void count();
	~LSH();

    int* add(int *indices, Node* node);
    int add(int indices, int tableId, Node* node);

    void getRequestIDs(int* hashes, int* hashes2, int *hashes3, int **requestIDs, int inputID);
};
