#include "DensifiedWtaHash.h"
#include <random>
#include <iostream>
#include <math.h>
#include <vector>
#include <climits>
#include <algorithm>
#include <map>
#include "Config.h"
using namespace std;

DensifiedWtaHash::DensifiedWtaHash(int numHashes, int K, int L, int noOfBitsToHash)
{

    _numhashes = numHashes;
    _rangePow = noOfBitsToHash;
    _K = K;
    _L = L;

    std::random_device rd;
    std::mt19937 gen(rd());

    _permute = _L * ceil(_K * binsize * 1.0 / noOfBitsToHash);
    //this only works if  _K*binsize < noOfBitsToHash,
    // the following permutation is also limited to this situation

    int* n_array = new int[_rangePow];
    _indices = new int[_rangePow*_permute];
    _pos = new int[_rangePow*_permute];

    for (int i = 0; i < _rangePow; i++) {
        n_array[i] = i;
    }
    for (int p=0; p<_permute ;p++) {
        std::shuffle(n_array, n_array+_rangePow, rd);

        //this only works if  _K*binsize < noOfBitsToHash,
        // the following permutation is also limited to this situation
        for (int j = 0; j < _K * binsize; j++){
            _indices[p * _rangePow + n_array[j]] = (p * _rangePow + j) / binsize;
            _pos[p * _rangePow + n_array[j]] = (p * _rangePow + j) % binsize;
        }
        for (int j = _K * binsize; j < _rangePow; j++){
            _indices[p * _rangePow + n_array[j]] = -1;
            _pos[p * _rangePow + n_array[j]] = -1;
            //-1 means the rest elements are not sampled
        }
    }
    delete [] n_array;

    _lognumhash = log2(numHashes);
    std::uniform_int_distribution<> dis(1, INT_MAX);

    _randa = dis(gen);
    if (_randa % 2 == 0)
        _randa++;
    _randHash = new int[2];
    _randHash[0] = dis(gen);
    if (_randHash[0] % 2 == 0)
        _randHash[0]++;
    _randHash[1] = dis(gen);
    if (_randHash[1] % 2 == 0)
        _randHash[1]++;

}

int * DensifiedWtaHash::getHashEasy(float* data, int dataLen, int topk)
{

    // binsize is the number of times the range is larger than the total number of hashes we need.

    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (size_t i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    for (int p = 0; p < _permute; p++){
        int bin_index = p * _rangePow;
        for ( int i = 0; i < dataLen; i++){
            int inner_index = bin_index + i;
            int binid = _indices[inner_index];
            if (binid == -1){
                // this element is not sampled,go to next one
                continue;
            }
            // 保护边界：只对合法 bin 进行更新
            if (binid >= 0 && binid < _numhashes) {
                float loc_data = data[i];

                // oblivious 版本的 “if (values[binid] < loc_data)”
                int updateFlag = obl::OblLt(values[binid], loc_data);    // 1 表示需要更新，否则 0

                // 按 flag 选择新旧值，不用普通 if
                float newVal = obl::OblChoose(updateFlag, loc_data, values[binid]);
                int   newHash = obl::OblChoose(updateFlag, _pos[inner_index], hashes[binid]);

                values[binid] = newVal;
                hashes[binid] = newHash;
            }
        }
    }

    for (size_t i = 0; i < _numhashes; i++)
    {
        int next = hashes[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes-1);

            next = hashes[index];
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] values;
    return hashArray;
}

int * DensifiedWtaHash::getHash(int* indices, float* data, int dataLen)
{

    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (size_t i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    for (int p = 0; p < _permute; p++){
        int bin_base = p * _rangePow;
        for (int i = 0; i < dataLen; i++){
            int dim = indices[i];
            int indexOffset = bin_base + dim;
            int binid = _indices[indexOffset];
            if (binid == -1){
                continue;
            }else{
                if (binid >= 0 && binid < _numhashes) {
                    float v = data[i];

                    int updateFlag = obl::OblLt(values[binid], v);

                    float newVal  = obl::OblChoose(updateFlag, v,                 values[binid]);
                    int   newHash = obl::OblChoose(updateFlag, _pos[indexOffset], hashes[binid]);

                    values[binid] = newVal;
                    hashes[binid] = newHash;
                }
            }
        }
    }

    for (size_t i = 0; i < _numhashes; i++)
    {
        int next = hashes[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes-1);

            next = hashes[index];
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] values;
    return hashArray;
}

int * DensifiedWtaHash::getHash2(int* indices, float* data, int dataLen)
{

    int *hashes = new int[_numhashes];
    int *hashes2 = new int[_numhashes];
    float *values = new float[_numhashes];
    float *values2 = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (size_t i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
        hashes2[i] = INT_MIN;
        values2[i] = INT_MIN;
    }

    // 主循环：oblivious 维护 top-2
    for (int p = 0; p < _permute; ++p) {
        int bin_base = p * _rangePow;

        for (int i = 0; i < dataLen; ++i) {
            int dim = indices[i];
            int indexOffset = bin_base + dim;
            int binid = _indices[indexOffset];

            if (binid == -1) {
                continue;
            }
            if (binid < 0 || binid >= _numhashes) {
                continue;
            }

            float v   = data[i];
            int   pos = _pos[indexOffset];

            float old1  = values[binid];
            float old2  = values2[binid];
            int   oldh1 = hashes[binid];
            int   oldh2 = hashes2[binid];

            // v 是否比当前 top1 / top2 更大（返回 0/1）
            int gtTop1 = obl::OblLt(old1, v);
            int gtTop2 = obl::OblLt(old2, v);

            int f1 = gtTop1;                // 升到第一
            int f2 = (1 - gtTop1) * gtTop2; // 只在没升第一时，才可能升第二

            // Stage 1: 处理“升第一”的情况
            float new1  = obl::OblChoose(f1, v,    old1);
            float new2  = obl::OblChoose(f1, old1, old2);
            int   newh1 = obl::OblChoose(f1, pos,  oldh1);
            int   newh2 = obl::OblChoose(f1, oldh1, oldh2);

            // Stage 2: 处理“升第二”的情况（在 f1=0 时才可能为 1）
            float final2  = obl::OblChoose(f2, v,   new2);
            int   finalh2 = obl::OblChoose(f2, pos, newh2);

            values[binid]  = new1;
            hashes[binid]  = newh1;
            values2[binid] = final2;
            hashes2[binid] = finalh2;
        }
    }

    for (size_t i = 0; i < _numhashes; i++)
    {
        int next = hashes2[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes2[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes-1);

            next = hashes2[index];
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] hashes2;
    delete[] values;
    delete[] values2;
    return hashArray;
}

int * DensifiedWtaHash::getHash3(int* indices, float* data, int dataLen)
{

    int *hashes = new int[_numhashes];
    int *hashes2 = new int[_numhashes];
    int *hashes3 = new int[_numhashes];
    float *values = new float[_numhashes];
    float *values2 = new float[_numhashes];
    float *values3 = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (size_t i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
        hashes2[i] = INT_MIN;
        values2[i] = INT_MIN;
        hashes3[i] = INT_MIN;
        values3[i] = INT_MIN;
    }

    // 主循环：oblivious 维护 top-3
    for (int p = 0; p < _permute; ++p) {
        int bin_base = p * _rangePow;

        for (int i = 0; i < dataLen; ++i) {
            int dim = indices[i];
            int indexOffset = bin_base + dim;
            int binid = _indices[indexOffset];

            if (binid == -1) {
                continue;
            }
            if (binid < 0 || binid >= _numhashes) {
                continue;
            }

            float v   = data[i];
            int   pos = _pos[indexOffset];

            float old1  = values[binid];
            float old2  = values2[binid];
            float old3  = values3[binid];
            int   oldh1 = hashes[binid];
            int   oldh2 = hashes2[binid];
            int   oldh3 = hashes3[binid];

            int gt1 = obl::OblLt(old1, v);
            int gt2 = obl::OblLt(old2, v);
            int gt3 = obl::OblLt(old3, v);

            int f1 = gt1;
            int f2 = (1 - gt1) * gt2;
            int f3 = (1 - gt1) * (1 - gt2) * gt3;

            // Stage 1: 先处理“升第一”的情况
            float v1 = obl::OblChoose(f1, v,    old1);
            float v2 = obl::OblChoose(f1, old1, old2);
            float v3 = obl::OblChoose(f1, old2, old3);
            int   h1 = obl::OblChoose(f1, pos,  oldh1);
            int   h2 = obl::OblChoose(f1, oldh1, oldh2);
            int   h3 = obl::OblChoose(f1, oldh2, oldh3);

            // Stage 2: 再处理“升第二”的情况
            float v2b = obl::OblChoose(f2, v,   v2);
            float v3b = obl::OblChoose(f2, v2,  v3);
            int   h2b = obl::OblChoose(f2, pos, h2);
            int   h3b = obl::OblChoose(f2, h2,  h3);

            // Stage 3: 最后处理“升第三”的情况
            float v3c = obl::OblChoose(f3, v,   v3b);
            int   h3c = obl::OblChoose(f3, pos, h3b);

            values[binid]  = v1;
            hashes[binid]  = h1;
            values2[binid] = v2b;
            hashes2[binid] = h2b;
            values3[binid] = v3c;
            hashes3[binid] = h3c;
        }
    }

    for (size_t i = 0; i < _numhashes; i++)
    {
        int next = hashes3[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes3[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes-1);

            next = hashes3[index];
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] hashes2;
    delete[] hashes3;
    delete[] values;
    delete[] values2;
    delete[] values3;
    return hashArray;
}

int DensifiedWtaHash::getRandDoubleHash(int binid, int count) {
    unsigned int tohash = ((binid + 1) << 6) + count;
    return (_randHash[0] * tohash << 3) >> (32 - _lognumhash); // _lognumhash needs to be ceiled.
}

DensifiedWtaHash::~DensifiedWtaHash()
{
    delete[] _randHash;
    delete[] _indices;
    delete[] _pos;
    _randHash = nullptr;
    _indices = nullptr;
    _pos = nullptr;
}