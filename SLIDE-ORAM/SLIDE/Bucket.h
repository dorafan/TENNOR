#pragma once
#include "Config.h"

class Bucket
{
private:

public:
    int *arr;
    int isInit = -1;
    int index = 0;
    int _counts = 0;

    int oram_ID = 0;

	Bucket();
	int add(int id);
	int retrieve(int index);
	int * getAll();
	int getTotalCounts();
	int getSize();
	~Bucket();

    std::string serialize() const;                
    void deserialize(const std::string& buffer);  
};

