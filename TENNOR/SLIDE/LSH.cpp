#include <iostream>
#include <unordered_map>
#include "LSH_New.h"
#include <climits>
#include "Config.h"
#include <chrono>
#pragma once

using namespace std;

LSH::LSH(int K, int L, int RangePow)
{
	_K = K;
	_L = L;
	_RangePow = RangePow;
	_bucket = new Bucket*[L];

	for (int i = 0; i < L; i++)
	{
		_bucket[i] = new Bucket[1 << _RangePow];
        for (int j = 0; j < (1 << _RangePow); j++) {
            _bucket[i][j]._bucID = j;
        }
	}

	rand1 = new int[_K*_L];

	std::random_device rd;
	 std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, INT_MAX);

	for (int i = 0; i < _K*_L; i++)
	{
		rand1[i] = dis(gen);
		if (rand1[i] % 2 == 0)
			rand1[i]++;
	}

}

void LSH::clear()
{
    for (int i = 0; i < _L; i++)
    {
    	delete[] _bucket[i];
    	_bucket[i] = new Bucket[1 << _RangePow];

    }
}

void LSH::count()
{
	for (int j=0; j<_L;j++) {
		int total = 0;
		for (int i = 0; i < 1 << _RangePow; i++) {
			if (_bucket[j][i].getSize()!=0) {
				cout <<_bucket[j][i].getSize() << " ";
			}
			total += _bucket[j][i].getSize();
		}
		cout << endl;
		cout <<"TABLE "<< j << "Total "<< total << endl;
	}
}

int* LSH::hashesToIndex(int * hashes)
{

	int * indices = new int[_L];
	for (int i = 0; i < _L; i++)
	{
		unsigned int index = 0;

		for (int j = 0; j < _K; j++)
		{

			if (HashFunction==4){
				unsigned int h = hashes[_K*i + j];
				index += h<<(_K-1-j);
			}else if (HashFunction==1 | HashFunction==2){
                unsigned int h = hashes[_K*i + j];
                index += h<<((_K-1-j)*(int)floor(log2(binsize)));

            }else {
                unsigned int h = rand1[_K*i + j];
                h *= rand1[_K * i + j];
                h ^= h >> 13;
                h ^= rand1[_K * i + j];
                index += h * hashes[_K * i + j];
            }
		}
		if (HashFunction==3) {
			index = index&((1<<_RangePow)-1);
		}
		indices[i] = index;
	}

	return indices;
}

int* LSH::add(int *indices, Node* node)
{
    int * secondIndices = new int[_L];
    for (int i = 0; i < _L; i++)
    {
        secondIndices[i] = _bucket[i][indices[i]].add(node);
    }

    return secondIndices;
}

int LSH::add(int tableId, int indices, Node* node)
{
    int secondIndices = _bucket[tableId][indices].add(node);
    return secondIndices;
}

int LSH::retrieve(int table, int indices, int bucket)
{
	return _bucket[table][indices].retrieve(bucket);
}

void LSH::getRequestIDs(int* hashes, int* hashes2, int *hashes3, int ** requestIDs, int inputID) {
//    Max-2 + Max-3 +      pair-Max-2      + Pair-Max-3            +       pair-Max-2-3       + triplet-Max-2 + triplet-Max-3 + triplet-Max-2-3
//      K   +   K   + _K * (_K - 1) / (2)  + _K * (_K - 1) / (2)   + _K * (_K - 1) / (2) * 2  +  _K * (_K - 1) * (_K - 2) / (3*2) * (2*2*2)
    int bucsNew = _K * (_K - 1) * (_K - 2) / (3*2) * (2*2*2) + _K * (_K - 1) / (2) * (2*2) + _K + _K + 1;

    int origin = hashesToIndex(hashes)[0];

    requestIDs[inputID][0] = origin;

    // Precompute the shifts based on _K and binsize
    std::vector<int> shifts(_K);
    for (int k = 0; k < _K; ++k) {
        shifts[k] = ((_K - 1 - k) * (int) floor(log2(binsize)));
    }

    for (int i = 0; i < _L; i++)
    {
        int index = 0;
        int count = 1;

        for (int j = 0; j < _K; j++)
        {
            for (int l = j+1; l < _K; l++){
                for (int m = l + 1; m < _K; m++){
                    if (HashFunction==1 | HashFunction==2){
                        int remain = 0;
                        for (int k = 0; k < _K; k++){
                            if (k != j && k != l && k != m) {
                                int h = hashes[_K * i + k];
                                remain += h << ((_K - 1 - k) * (int) floor(log2(binsize)));
                            }
                        }

                        for (int choice = 0; choice < 8; ++choice) {
                            int mpindex = remain;

                            int source_j = (choice & 1) ? hashes3[i * _K + j] : hashes2[i * _K + j];
                            int source_l = (choice & 2) ? hashes3[i * _K + l] : hashes2[i * _K + l];
                            int source_m = (choice & 4) ? hashes3[i * _K + m] : hashes2[i * _K + m];

                            mpindex += source_j << shifts[j];
                            mpindex += source_l << shifts[l];
                            mpindex += source_m << shifts[m];

                            requestIDs[inputID][count] = mpindex;
                            count++;
                        }

                    }
                }
            }

        }
        for (int j = 0; j < _K; j++)
        {
            for (int l = j+1; l < _K; l++){
                if (HashFunction==1 | HashFunction==2){
                    int remain = 0;
                    for (int k = 0; k < _K; k++){
                        if (k != j && k != l) {
                            int h = hashes[_K * i + k];
                            remain += h << ((_K - 1 - k) * (int) floor(log2(binsize)));
                        }
                    }

                    for (int choice = 0; choice < 4; ++choice) {
                        int mpindex = remain;

                        int source_j = (choice & 1) ? hashes3[i * _K + j] : hashes2[i * _K + j];
                        int source_l = (choice & 2) ? hashes3[i * _K + l] : hashes2[i * _K + l];

                        mpindex += source_j << shifts[j];
                        mpindex += source_l << shifts[l];

                        requestIDs[inputID][count] = mpindex;
                        count++;
                    }

                }

            }

        }
        for (int j = 0; j < _K; j++)
        {
            if (HashFunction==1 | HashFunction==2){
                int remain = 0;
                for (int k = 0; k < _K; k++){
                    if (k != j) {
                        int h = hashes[_K * i + k];
                        remain += h << ((_K - 1 - k) * (int) floor(log2(binsize)));
                    }
                }

                for (unsigned int choice = 0; choice < 2; ++choice) {
                    int mpindex = remain;

                    int source_j = (choice & 1) ? hashes3[i * _K + j] : hashes2[i * _K + j];

                    mpindex += source_j << shifts[j];

                    requestIDs[inputID][count] = mpindex;
                    count++;
                }

            }else {
                unsigned int h = rand1[_K*i + j];
                h *= rand1[_K * i + j];
                h ^= h >> 13;
                h ^= rand1[_K * i + j];
                index += h * hashes[_K * i + j];
            }

        }

    }

    shifts.clear();

}

LSH::~LSH()
{
	delete [] rand1;
	 for (size_t i = 0; i < _L; i++)
	 {
	 	delete[] _bucket[i];
	 }
	 delete[] _bucket;
}

