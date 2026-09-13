


#include <algorithm>
#include "Requester.h"

#include "Bucket.h"
#include <type_traits>
#include "obl_primitives.h"
























struct BucketHeaderScalarBlock {
    decltype(Bucket::isInit)  isInit;
    decltype(Bucket::index)   index;
    decltype(Bucket::_counts) _counts;
    decltype(Bucket::_bucID)  _bucID;
};

struct RequestBucOwnScalarBlock {
    decltype(RequestBuc::_inputRank) _inputRank;
    decltype(RequestBuc::tag)        tag;
    decltype(RequestBuc::bin_id)     bin_id;
    decltype(RequestBuc::oht_pos)    oht_pos;
};


int RequestBuc::getRequestTag()  { return _inputRank; }

void RequestBuc::setRequestTag(int tag) { _inputRank = tag; }

RequestBuc::RequestBuc(int dim, NodeType type, int layerID,   int batchsize) : Bucket(), _inputRank(0)
{

















}



void RequestBuc::obl_copy_reqbuc(bool pred, RequestBuc* source, RequestBuc* dest)
{

    if (!source || !dest) {

        return;
    }

     using snoopy::obl::ObliviousBytesAssign;













































    {
        const void* src = static_cast<const void*>(std::addressof(source->_inputRank));
        const void* dst = static_cast<const void*>(std::addressof(dest->_inputRank));
        void*       out = static_cast<void*>(std::addressof(dest->_inputRank));

        ObliviousBytesAssign(pred,
                            sizeof(RequestBucOwnScalarBlock),
                            src,
                            dst,
                            out);
    }


    {
        const void* src = static_cast<const void*>(std::addressof(source->isInit));
        const void* dst = static_cast<const void*>(std::addressof(dest->isInit));
        void*       out = static_cast<void*>(std::addressof(dest->isInit));

        ObliviousBytesAssign(pred,
                            sizeof(BucketHeaderScalarBlock),
                            src,
                            dst,
                            out);
    }













    const int bucketSize = BUCKETSIZE;



    for (int i = 0; i < bucketSize; ++i)
    {
        if(!dest->arr[i] && !source->arr[i]){
            continue;
        }

        if (dest->arr[i] == nullptr )
        {

            dest->arr[i] = new Node();
        }


        Node* sourceNode = source->arr[i] ? source->arr[i] : nullptr;
        Node* destNode = dest->arr[i];

        if (sourceNode && destNode)
        {

            Node::obl_copy_node(pred, sourceNode, destNode);
        }








    }
}

void RequestBuc::obl_copy_buc_to_reqbuc(bool pred, const Bucket* source, RequestBuc* dest) {

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


    for (int i = 0; i < bucketSize; ++i)
    {

       if (dest->arr[i] == nullptr )
       {
           dest->arr[i] = new Node();
       }


        Node* sourceNode = source->arr[i];
        Node* destNode = dest->arr[i];

        if (sourceNode && destNode)
        {

            Node::obl_copy_node(pred, sourceNode, destNode);
        }







    }
}


void RequestBuc::obl_copy_reqbuc_to_buc(bool pred,
                                        const RequestBuc* source,
                                        Bucket* dest)
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

    for (int i = 0; i < bucketSize; ++i)
    {
        Node* srcNode = source->arr[i];
        Node* dstNode = dest->arr[i];



        if (!dstNode && srcNode) {
            dest->arr[i] = new Node();
            dstNode = dest->arr[i];
        }


        if (srcNode && dstNode) {
            Node::obl_copy_node(pred, srcNode, dstNode);
        }


    }
}



Requester::Requester(int totalReqs) : _totalReqs(totalReqs) {
    _reqTable = new RequestBuc*[_totalReqs];
}

Requester::~Requester() {
    for (int i = 0; i < _currentReqIndex; ++i) {
        if (_reqTable[i] != nullptr)
            delete _reqTable[i];
            _reqTable[i] = nullptr;
    }
    delete[] _reqTable;
    _reqTable = nullptr;
}

void Requester::generateRequestBucs(int* requestIDs, int numRequests, int inputID, int dim, NodeType type, int layerID,  int batchsize) {
    for (int i = 0; i < numRequests; ++i) {
        _reqTable[_currentReqIndex] = new RequestBuc(dim,  type,  layerID,   batchsize);
        _reqTable[_currentReqIndex]->_inputRank = inputID;
        _reqTable[_currentReqIndex]->_bucID = requestIDs[i];
        _currentReqIndex ++;

    }
}




void Requester::sortByIDBitonicPtr()
{


    uint32_t n = _currentReqIndex;

    auto cmp = [](const RequestBuc* a, const RequestBuc* b) {

        return obl::OblLt(a->_bucID, b->_bucID);
    };



    RequestBuc scratch(*_reqTable[0]);





    o_sort_ReqBucPtr_Parallel(_reqTable, 0, n, cmp);



}





void Requester::sortByInputRank() {
    std::sort(_reqTable, _reqTable + _currentReqIndex, [](const RequestBuc* a, const RequestBuc* b) {
        return a->_inputRank < b->_inputRank;
    });
}











































namespace {



constexpr int kTagDummyBit = 1 << 0;
}

void Requester::obl_mergeSameBucID(int &totalReqMerge) {










    for (int i = 0; i < totalReqMerge - 1; i++) {
        int j = i + 1;
        RequestBuc* mergedBuc = _reqTable[j];
        RequestBuc* foldedBuc = _reqTable[i];

        bool pred = obl::OblEq(foldedBuc->_bucID, mergedBuc->_bucID);



        mergedBuc->obl_mergeWith(foldedBuc, pred);


        foldedBuc->tag = obl::OblChoose(pred,
                                        foldedBuc->tag | kTagDummyBit,
                                        foldedBuc->tag);
    }











}

void RequestBuc::obl_mergeWith(RequestBuc *other, bool pred) {






    for (int i = 0; i < BUCKETSIZE; i++) {
        Node * mergeTo   = arr[i];
        Node * mergeFrom = other->arr[i];

        bool sameNeuron = obl::OblEq(mergeFrom->_IDinLayer, mergeTo->_IDinLayer);
        bool sameDim    = obl::OblEq(mergeTo->_dim, mergeFrom->_dim);

        bool fold = pred & sameNeuron & sameDim;

        for (int d = 0; d < mergeTo->_dim; d++) {
            float toVal   = mergeTo->_t[d];
            float fromVal = mergeFrom->_t[d];
            mergeTo->_t[d]   = obl::OblChoose(fold, toVal + fromVal, toVal);
            mergeFrom->_t[d] = obl::OblChoose(fold, 0.0f, fromVal);
        }
        float toBias   = mergeTo->_tbias;
        float fromBias = mergeFrom->_tbias;
        mergeTo->_tbias   = obl::OblChoose(fold, toBias + fromBias, toBias);
        mergeFrom->_tbias = obl::OblChoose(fold, 0.0f, fromBias);
    }
}




void Requester::sortByID() {
    std::sort(_reqTable, _reqTable + _currentReqIndex, [](const RequestBuc* a, const RequestBuc* b) {
        return a->_bucID < b->_bucID;
    });
}

void Requester::mergeSameBucID(int& totalReqMerge) {
    int i = 0;
    while (i < totalReqMerge) {
        int currentBucID = _reqTable[i]->_bucID;
        int start = i;

        while (i < totalReqMerge && _reqTable[i]->_bucID == currentBucID) {
            i++;
        }
        int end = i;
        int count = end - start;

        if (count > 1) {


            mergeRequestBucs(start, end);
        }
    }



    removeNullPointers(totalReqMerge);
}

void Requester::mergeRequestBucs(int start, int end) {
    RequestBuc* mergedBuc = _reqTable[start];

    for (int i = start + 1; i < end; ++i) {
        mergedBuc->mergeWith(_reqTable[i]);
        delete _reqTable[i];
        _reqTable[i] = nullptr;
    }
}

void Requester::removeNullPointers(int &totalReqMerge) {
    int writeIndex = 0;
    for (int readIndex = 0; readIndex < totalReqMerge; ++readIndex) {
        if (_reqTable[readIndex] != nullptr) {
            _reqTable[writeIndex++] = _reqTable[readIndex];
        }
    }
    totalReqMerge = writeIndex;


    _currentReqIndex = writeIndex;
}




void RequestBuc::mergeWith(const RequestBuc *other) {
    for(int i = 0; i < BUCKETSIZE; i++){
        Node * mergeTo= arr[i];
        Node * mergeFrom = other->arr[i];

        if (mergeFrom->_IDinLayer == mergeTo->_IDinLayer && mergeTo->_dim == mergeFrom->_dim) {
            for (int j = 0; j < mergeTo->_dim; j++) {
                mergeTo->_t[j] += mergeFrom->_t[j];
            }
            mergeTo->_tbias += mergeFrom->_tbias;
        }
    }
}