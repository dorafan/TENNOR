#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <vector>
#include <string.h>
#include "MurmurHash.h"

using namespace std;
class DensifiedWtaHash
{
private:
    int *_randHash, _randa, _numhashes, _rangePow,_lognumhash, *_indices, *_pos, _permute;
    int _K, _L;
public:
    DensifiedWtaHash(int numHashes, int K, int L, int noOfBitsToHash);
    int * getHash(int* indices, float* data, int dataLen);
    int * getHash2(int* indices, float* data, int dataLen);
    int * getHash3(int* indices, float* data, int dataLen);
    int getRandDoubleHash(int binid, int count);
    int * getHashEasy(float* data, int dataLen, int topK);
    ~DensifiedWtaHash();
};