#include <iostream>
#include "Bucket.h"

#pragma once









struct BucketHeaderScalarBlock {
    decltype(Bucket::isInit)  isInit;
    decltype(Bucket::index)   index;
    decltype(Bucket::_counts) _counts;
    decltype(Bucket::_bucID)  _bucID;
};



void Bucket::padding(int dim, NodeType type, int layerID, int batchsize)
{


    int counttmp = _counts;

    if (counttmp < BUCKETSIZE){
        for (int i = counttmp; i < BUCKETSIZE; i++){
            if (arr[i] == nullptr) {
                arr[i] = new Node(dim, type, layerID, batchsize);
                arr[i]->isDummy = 1;
            }
        }
    }


}





void Bucket::obl_copy_bucket(bool pred, const Bucket* source, Bucket* dest)
{

    if (!source || !dest) {
        return;
    }






    using snoopy::obl::ObliviousBytesAssign;
    const void* src = static_cast<const void*>(std::addressof(source->isInit));
    const void* dst = static_cast<const void*>(std::addressof(dest->isInit));
    void*       out = static_cast<void*>(std::addressof(dest->isInit));

    ObliviousBytesAssign(pred,
                         sizeof(BucketHeaderScalarBlock),
                         src,
                         dst,
                         out);


    const int bucketSize = BUCKETSIZE;




























    for (int i = 0; i < BUCKETSIZE; ++i) {
        Node* src = source->arr[i];
        Node*& dst = dest->arr[i];


        if (!src && !dst) {
            continue;
        }


        if (!dst) {
            dst = new Node();
        }

        if (src) {

            Node::obl_copy_node(pred, src, dst);
        } else {



            Node tmp_empty;
            Node::obl_copy_node(pred, &tmp_empty, dst);




        }
}

}



int Bucket::getTotalCounts()
{
    return _counts;
}


int Bucket::getSize()
{
    return _counts;
}

































int Bucket::add(Node* node) {


    if (FIFO) {
        isInit += 1;
        index = _counts & (BUCKETSIZE - 1);


        _counts++;

        if (arr[index] == nullptr) {

            arr[index] = new Node(*node);
        }
        else {
            if(_counts >= BUCKETSIZE){
                printf("an Node is trying to add to a full Bucket, bucket ID: %d,  Node ID : %d, bucket location: %d\n", _bucID, node->_IDinLayer, this->index);
            }
        }



        return index;


    }



















}


int Bucket::retrieve(int indice)
{
    if (indice >= BUCKETSIZE || arr[indice] == nullptr)
        return -1;
    return arr[indice]->_IDinLayer;
}











