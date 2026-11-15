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
                         out); // out == dst

    const int bucketSize = BUCKETSIZE;

    for (int i = 0; i < BUCKETSIZE; ++i) {
        Node* src = source->arr[i];
        Node*& dst = dest->arr[i];

        // 如果两边都是空，什么都不做
        if (!src && !dst) {
            continue;
        }

        // 确保 dst 至少存在一个 Node 对象，好让 obl_copy_node 有地方写
        if (!dst) {
            dst = new Node();
        }

        if (src) {
            // 正常从 src 拷到 dst
            Node::obl_copy_node(pred, src, dst);
        } else {
            // src == nullptr，逻辑上代表“空节点”
                       
            // 用一个默认 Node 覆盖 dst
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

    //FIFO
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

