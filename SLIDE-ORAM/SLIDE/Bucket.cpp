#include <iostream>
#include "Bucket.h"

#include <cstring>
#pragma once

Bucket::Bucket()
{
    isInit = -1;
    arr = new int[BUCKETSIZE]();
}

Bucket::~Bucket()
{
    delete[] arr;
}

int Bucket::getTotalCounts()
{
    return _counts;
}

int Bucket::getSize()
{
    return _counts;
}

int Bucket::add(int id) {

    //FIFO
    if (FIFO) {
        isInit += 1;
        int index = _counts & (BUCKETSIZE - 1);
        _counts++;
        arr[index] = id;
        return index;
    }
    //Reservoir Sampling
    else {
        _counts++;
        if (index == BUCKETSIZE) {
            int randnum = rand() % (_counts) + 1;
            if (randnum == 2) {
                int randind = rand() % BUCKETSIZE;
                arr[randind] = id;
                return randind;
            } else {
                return -1;
            }
        } else {
            arr[index] = id;
            int returnIndex = index;
            index++;
            return returnIndex;
        }
    }
}

int Bucket::retrieve(int indice)
{
    if (indice >= BUCKETSIZE)
        return -1;
    return arr[indice];
}

int * Bucket::getAll()
{
    if (isInit == -1)
        return NULL;
    if(_counts<BUCKETSIZE){
        arr[_counts]=-1;
    }
    return arr;
}

std::string Bucket::serialize() const {

    size_t dataSize = sizeof(isInit) + sizeof(index) + sizeof(_counts) + sizeof(int) * BUCKETSIZE;

    // Create a string with the target size, initialized with zeros
    std::string buffer(SERIAL_LEN, '\0');

    size_t offset = 0;

    std::memcpy(&buffer[offset], &isInit, sizeof(isInit));
    offset += sizeof(isInit);

    std::memcpy(&buffer[offset], &index, sizeof(index));
    offset += sizeof(index);

    std::memcpy(&buffer[offset], &_counts, sizeof(_counts));
    offset += sizeof(_counts);

    std::memcpy(&buffer[offset], arr, sizeof(int) * BUCKETSIZE);
    offset += sizeof(int) * BUCKETSIZE;

    // Padding is already zero-initialized due to the initial buffer size

    return buffer;
}

void Bucket::deserialize(const std::string& buffer) {
    size_t offset = 0;

    // Ensure the buffer is at least as big as the data we need to read
    size_t expectedSize = sizeof(isInit) + sizeof(index) + sizeof(_counts) + sizeof(int) * BUCKETSIZE;

    if (buffer.size() < expectedSize) {
        if (TEST) {
            std::cout << "BUCKET Expect Buffre Size : " << expectedSize << ",  real buffer size : " << buffer.size() << std::endl;
        }

    }

    std::memcpy(&isInit, &buffer[offset], sizeof(isInit));
    offset += sizeof(isInit);

    std::memcpy(&index, &buffer[offset], sizeof(index));
    offset += sizeof(index);

    std::memcpy(&_counts, &buffer[offset], sizeof(_counts));
    offset += sizeof(_counts);

    std::memcpy(arr, &buffer[offset], sizeof(int) * BUCKETSIZE);

    // Padding is ignored
}

