
#pragma once
#include "Config.h"

#include "Node.h"

#include "obl.h"







class Bucket
{
private:






public:
    int isInit = -1;
    int index = 0;
    int _counts = 0;

    int _bucID = 0;

    Node** arr = nullptr;

	int add(int id);
    int add(Node* node);
	int retrieve(int index);

	int getTotalCounts();
	int getSize();





    static void obl_copy_bucket(bool pred, const Bucket* source, Bucket* dest);
    void padding(int dim, NodeType type, int layerID, int batchsize);



    Bucket()
            : isInit(-1), index(0), _counts(0), _bucID(0), arr(new Node*[BUCKETSIZE]())
    {

        std::fill(arr, arr + BUCKETSIZE, nullptr);
    }



    virtual ~Bucket()
    {
        if (arr != nullptr) {
            for (int i = 0; i < BUCKETSIZE; i++){
                if (arr[i] != nullptr){
                    delete arr[i];
                }
            }
        }
        delete [] arr;

        arr = nullptr;

    }




    Bucket(const Bucket& other)
            : isInit(other.isInit), index(other.index), _counts(other._counts), _bucID(other._bucID), arr(new Node*[BUCKETSIZE]())
    {
        for (int i = 0; i < BUCKETSIZE; i++) {
            if (other.arr[i] != nullptr) {
                arr[i] = new Node(*other.arr[i]);
            } else {
                arr[i] = nullptr;
            }
        }
    }



    Bucket& operator=(const Bucket& other)
    {
        if (this == &other) return *this;


        if (arr != nullptr) {
            for (int i = 0; i < BUCKETSIZE; i++) {
                if (arr[i] != nullptr){
                    delete arr[i];
                    arr[i] = nullptr;
                }
            }

            delete [] arr;

            arr = nullptr;

        }

        arr = new Node*[BUCKETSIZE]();
        std::fill(arr, arr + BUCKETSIZE, nullptr);


        isInit = other.isInit;
        index = other.index;
        _counts = other._counts;
        _bucID = other._bucID;


        for (int i = 0; i < BUCKETSIZE; i++) {
            if (other.arr[i] != nullptr) {


                arr[i] = new Node(*other.arr[i]);

            } else {
                arr[i] = nullptr;
            }
        }

        return *this;
    }



    Bucket(Bucket&& other) noexcept
            : isInit(other.isInit), index(other.index), _counts(other._counts), _bucID(other._bucID), arr(other.arr)
    {






        other.arr = nullptr;

        other.isInit = -1;
        other.index = 0;
        other._counts = 0;
        other._bucID = 0;
    }


    Bucket& operator=(Bucket&& other) noexcept
    {
        if (this == &other) return *this;


        if (arr != nullptr) {
            for (int i = 0; i < BUCKETSIZE; i++) {
                if (arr[i] != nullptr){
                    delete arr[i];
                    arr[i] = nullptr;
                }
            }
        }
        delete [] arr;
        arr = nullptr;


        arr = other.arr;





        other.arr = nullptr;


        isInit = other.isInit;
        index = other.index;
        _counts = other._counts;
        _bucID = other._bucID;




        other.isInit = -1;
        other.index = 0;
        other._counts = 0;
        other._bucID = 0;

        return *this;
    }


};


