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
    decltype(RequestBuc::oht_pos)    oht_pos;   // Includes oht_pos for restoring the OHT layout.
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

        // Copy RequestBuc scalar fields, including oht_pos, as one block.
    {
        const void* src = static_cast<const void*>(std::addressof(source->_inputRank));
        const void* dst = static_cast<const void*>(std::addressof(dest->_inputRank));
        void*       out = static_cast<void*>(std::addressof(dest->_inputRank));

        ObliviousBytesAssign(pred,
                            sizeof(RequestBucOwnScalarBlock),
                            src,
                            dst,
                            out); // out == dst：in-place
    }

    // (B) 再拷贝 Bucket 头部的 4 个标量：从 &isInit 开始，连续覆盖 4 个 int
    {
        const void* src = static_cast<const void*>(std::addressof(source->isInit));
        const void* dst = static_cast<const void*>(std::addressof(dest->isInit));
        void*       out = static_cast<void*>(std::addressof(dest->isInit));

        ObliviousBytesAssign(pred,
                            sizeof(BucketHeaderScalarBlock),
                            src,
                            dst,
                            out); // out == dst：in-place
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

    // 只拷贝 Bucket 头部标量（4 个 int），从 &isInit 开始
    const void* src = static_cast<const void*>(std::addressof(source->isInit));
    const void* dst = static_cast<const void*>(std::addressof(dest->isInit));
    void*       out = static_cast<void*>(std::addressof(dest->isInit));

    ObliviousBytesAssign(pred,
                        sizeof(BucketHeaderScalarBlock),
                        src,
                        dst,
                        out); // out == dst：in-place

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
                        out); // out == dst：in-place

    // 3) 再 oblivious 拷贝 Node 数组
    const int bucketSize = BUCKETSIZE;

    for (int i = 0; i < bucketSize; ++i)
    {
        Node* srcNode = source->arr[i];
        Node* dstNode = dest->arr[i];

        // 如果 dest 这一格还没分配、而 source 有节点，就先给 dest 分配一个空 Node
        // 这样就不会把 nullptr 传给 Node::obl_copy_node
        if (!dstNode && srcNode) {
            dest->arr[i] = new Node();
            dstNode = dest->arr[i];
        }

        if (srcNode && dstNode) {
            Node::obl_copy_node(pred, srcNode, dstNode);
        }
        // 如果 srcNode 为 nullptr，那这一格什么都不做；
        // 如果两个都是 nullptr，更是直接跳过
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

    // 用第一个元素拷贝构造一个局部 scratch
    // 这样 scratch 里 arr[i] 一定都已经分配好了，obl_copy_reqbuc 不会遇到 nullptr
    RequestBuc scratch(*_reqTable[0]);

    o_sort_ReqBucPtr_Parallel(_reqTable, 0, n, cmp);

}

void Requester::sortByInputRank() {
    std::sort(_reqTable, _reqTable + _currentReqIndex, [](const RequestBuc* a, const RequestBuc* b) {
        return a->_inputRank < b->_inputRank;
    });
}

namespace {
// Mirrors Oht::kTagDummy bit; duplicated here to avoid a Requester.cpp -> Oht.h
// include cycle (Oht.h already includes Requester.h). Keep in sync if Oht's
// kTagDummy value ever changes (currently 1 << 0).
constexpr int kTagDummyBit = 1 << 0;
}

void Requester::obl_mergeSameBucID(int &totalReqMerge) {
    // Paper Algorithm 8 step 2: forward scan adjacent pairs (i, i+1). When their
    // _bucID matches, fold reqTable[i] (left) INTO reqTable[i+1] (right) and zero
    // reqTable[i]'s _t/_tbias, then mark reqTable[i] as dummy. Within each
    // same-bucket run, the rightmost entry accumulates the full sum; the others
    // become dummies with zero grad so subsequent Adam updates them without
    // contributing new gradient.
      
    // Caller must have already sorted by _bucID (sortByIDBitonicPtr). No `delete`,
    // no compaction: dummies are kept in-place, so the access pattern depends only
    // on the public count totalReqMerge.
    for (int i = 0; i < totalReqMerge - 1; i++) {
        int j = i + 1;
        RequestBuc* mergedBuc = _reqTable[j];   // accumulator (right)
        RequestBuc* foldedBuc = _reqTable[i];   // donor (left), becomes dummy iff pred

        bool pred = obl::OblEq(foldedBuc->_bucID, mergedBuc->_bucID);

        // FoldIntoPrev: mergedBuc->_t[] += foldedBuc->_t[];  foldedBuc->_t[] = 0
        // (and same for _tbias). Always walks all slots/dims; guarded by OblChoose.
        mergedBuc->obl_mergeWith(foldedBuc, pred);

        // MarkDummy: foldedBuc->tag |= kTagDummyBit  iff pred=true.
        foldedBuc->tag = obl::OblChoose(pred,
                                        foldedBuc->tag | kTagDummyBit,
                                        foldedBuc->tag);
    }

    // NOTE: paper Algorithm 8 step 3 ("OblSort by isDummy") is intentionally
    // omitted here. The next stage rebuilds the OHT for write-back, and OHT's
    // own build_pass cmp2 already encodes the kTagDummy bit into its sort key
    // (real entries first within each bin, dummies after). A separate
    // OblSort by isDummy in this function would be wasted work.
      
    // For the merge to be safe end-to-end, Oht::lookupLSH_write must AND the
    // existing match predicate with `!(tag & kTagDummyBit)` so dummy entries
    // do a fake write (oblivious copy with pred=false) rather than overwriting
    // the LSH bucket.
}

void RequestBuc::obl_mergeWith(RequestBuc *other, bool pred) {
    // For each Node slot in the bucket, fold `other`'s grad into `this` and zero
    // `other`'s grad iff (pred && same neuron && same dim). Same-bucket invariant
    // guarantees the neuron/dim equality whenever pred=true, but we still
    // oblivious-AND the checks so a violation cannot leak via short-circuit.
    // mergeTo->_dim is the layer dimension (public), so the inner dim loop bound
    // is also public.
    for (int i = 0; i < BUCKETSIZE; i++) {
        Node * mergeTo   = arr[i];
        Node * mergeFrom = other->arr[i];

        bool sameNeuron = obl::OblEq(mergeFrom->_IDinLayer, mergeTo->_IDinLayer);
        bool sameDim    = obl::OblEq(mergeTo->_dim, mergeFrom->_dim);
        // Bitwise & (no short-circuit) so all three operands are always evaluated.
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