#pragma once
#include "Node.h"
#include "WtaHash.h"
#include "DensifiedMinhash.h"
#include "srp.h"
#include "LSH_New.h"
#include "DensifiedWtaHash.h"
#include "cnpy.h"
#include <fstream>
#include <cstdlib>

#include <random>
#include <chrono>
#include <cassert>
#include <cmath>

#include "Requester.h"

#include "Oht.h"

#include "omp.h"

using namespace std;

class Layer
{
private:

    Node** _Nodes;

public:
    std::vector<Node> _AllNodes {};
    std::vector<Node> _overflows {};

    NodeType _type;
    float* _normalizationConstants;
    int*_inputIDs; //needed for SOFTMAX

    int _K, _L, _previousLayerNumOfNodes, _batchsize;

    int _RangePow;
    int _layerID, _noOfActive;
    int _noOfNodes;
    float* _weights;
    float* _adamAvgMom;
    float* _adamAvgVel;
    float* _bias;
    LSH *_hashTables;
    WtaHash *_wtaHasher;
    DensifiedMinhash *_MinHasher;
    SparseRandomProjection *_srp;
    DensifiedWtaHash *_dwtaHasher;
    int * _binids;
    Layer(int _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity, float* weights=NULL, float* bias=NULL, float *adamAvgMom=NULL, float *adamAvgVel=NULL);
    Node* getNodebyID(int nodeID);
    Node** getAllNodes();
    float getNomalizationConstant(int inputID);
    int queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter, bool tmpRehash, bool tmpRebuild);
    void saveWeights(string file);
    void updateTable();
    void updateRandomNodes();

    static bool myfunction(float i, float j);
    static bool myfunction2(float i, float j);

    Requester* _requester;
    int addtoHashTable(Node* node);

    void query(int ***activenodesperlayer, float *** activeValuesperlayer, int **sizes, int layerID, int inputID, float Sparsity, int **requestIDs);

    int ComputeActivation_FirstLayer(int **inputIndices, float **inputValues, int *lengths, int ***activenodesperlayer, float ***activeValuesperlayer, int **sizes, int layerID, int inputID, float Sparsity);

    Oht* _oht;

    ~Layer();

    bool isLSHinit = false;

// 1) swapNode: a custom swap that uses move semantics
    static void swapNode(Node &lhs, Node &rhs) noexcept {
        if (&lhs == &rhs) {
            return;
        }
        Node temp(std::move(lhs));
        lhs = std::move(rhs);
        rhs = std::move(temp);

    }

    // pred == 1 时做交换，pred == 0 时逻辑上“不交换”。
    // 无论 pred 是 0 还是 1，都会对 a/b/tmp 做同样模式的访问。
    static void obl_swapNode(Node &a, Node &b, int pred) noexcept {
        pred = pred ? 1 : 0;

        // 临时 Node，用默认构造就行；obl_copy_node(0, ...) 必须是 no-op
        Node tmp(a);

        Node::obl_copy_node(pred, &a, &tmp);
        Node::obl_copy_node(pred, &b, &a);
        Node::obl_copy_node(pred, &tmp, &b);
    }

// 2) Greatest Power of Two <= x
//    e.g., if x=10 => returns 8; if x=16 => returns 16
    static uint32_t greatestPowerOfTwoLessOrEqual(uint32_t x) {
        if (x < 1) return 0;
        uint32_t p = 1;
        while ((p << 1) <= x) {
            p <<= 1;
        }
        return p;
    }

// 3) XOR-based Imperative Bitonic Sort for power-of-two array size
//    - Sorts 'arr' in ascending order using the comparator 'cmp'.
//    - Must only be called if arr.size() is exactly a power-of-two!
    template<typename Compare>
    void imperativeXorBitonicSort(std::vector<Node> &arr, Compare cmp)
    {
        size_t n = arr.size();
        for (size_t k = 2; k <= n; k <<= 1) {
            for (size_t j = k >> 1; j > 0; j >>= 1) {
                for (size_t i = 0; i < n; i++) {
                    size_t ij = i ^ j;
                    if (ij > i) {
                        // Decide ascending or descending half
                        // If (i & k) == 0 => ascending half
                        bool ascendingHalf = ((i & k) == 0);

                        int pred;
                        if (ascendingHalf) {
                            // 升序区：需要交换当且仅当 arr[i] > arr[ij]
                            pred = static_cast<int>(cmp(arr[ij], arr[i]));
                        } else {
                            // 降序区：需要交换当且仅当 arr[i] < arr[ij]
                            pred = static_cast<int>(cmp(arr[i], arr[ij]));
                        }

                        // 无论 pred 是 0 还是 1，都走相同的访问模式
                        obl_swapNode(arr[i], arr[ij], pred);
                    }
                }
            }
        }
    }

// 4) Imperative bitonic-like merge for non-power-of-two approach
//    - Merges [low..low+len) in ascending order.
//    - This routine expects each sub-block is already ascending
//      but arr is in a "bitonic" configuration. The result is ascending.
    template<typename Compare>
    void imperativeXorBitonicMerge(std::vector<Node> &arr,
                                   uint32_t low, uint32_t len,
                                   Compare cmp)
    {
        if (len < 2) return;

        // We need the next power-of-two >= len
        // So we can perform log2 passes.
                            
        uint32_t size = 1;
        while (size < len) size <<= 1;

        for (uint32_t step = size >> 1; step > 0; step >>= 1) {
            // 注意：这里用的是相对索引 i' ∈ [0, len)
            for (uint32_t iRel = 0; iRel < len; ++iRel) {
                uint32_t jRel = iRel ^ step;

                // 保证只处理 [0,len) 内部的 pair，且每对只处理一次
                if (jRel > iRel && jRel < len) {
                    uint32_t i = low + iRel;
                    uint32_t j = low + jRel;

                    // 升序：swap if arr[i] > arr[j]
                    int pred = static_cast<int>(cmp(arr[j], arr[i]));  // arr[i] > arr[ij] 时为 1
                    obl_swapNode(arr[i], arr[j], pred);
                }
            }
        }
    }

// 5) XOR-based Bitonic Sort for Non-Power-of-Two array
//    - Splits array at greatest power-of-two < len,
//      sorts sub-block [low..low+m) ascending, then reverses => effectively descending,
//      recursively sorts [low+m..low+len) ascending,
//      then merges them in ascending order.
    template<typename Compare>
    void xorBitonicSort(std::vector<Node> &arr,
                        uint32_t low, uint32_t len,
                        Compare cmp)
    {
        if (len <= 1) return;

        // largest power-of-two <= len
        uint32_t m = greatestPowerOfTwoLessOrEqual(len);
        if (m == len) {
            // Perfect power-of-two => call the direct XOR-based bitonic sort
            // on the subarray [low..low+len).

            std::vector<Node> tmp(arr.begin() + low, arr.begin() + low + len);
            imperativeXorBitonicSort(tmp, cmp);
            for (uint32_t i = 0; i < len; i++) {
                arr[low + i] = std::move(tmp[i]);
            }
        } else {

            {
                std::vector<Node> tmp(arr.begin() + low, arr.begin() + low + m);
                imperativeXorBitonicSort(tmp, cmp);
                // reverse => effectively descending
                std::reverse(tmp.begin(), tmp.end());
                for (uint32_t i = 0; i < m; i++) {
                    arr[low + i] = std::move(tmp[i]);
                }
            }
            xorBitonicSort(arr, low + m, len - m, cmp);
            imperativeXorBitonicMerge(arr, low, len, cmp);
        }
    }

    template<typename Compare>
    static inline void imperativeXorBitonicSort_Parallel(std::vector<Node> &arr, Compare cmp)
    {
        const std::size_t n = arr.size();
        if (n <= 1) return;

        for (std::size_t k = 2; k <= n; k <<= 1) {
            for (std::size_t j = (k >> 1); j > 0; j >>= 1) {

                // 这一层的每个 i-比较对是互不重叠的，可以并行

    #pragma omp parallel for schedule(static)

                for (std::ptrdiff_t ti = 0; ti < static_cast<std::ptrdiff_t>(n); ++ti) {
                    std::size_t i  = static_cast<std::size_t>(ti);
                    std::size_t ij = i ^ j;

                    if (ij > i) {
                        // 升序 or 降序是公开信息（只依赖 i 和 k），不影响 oblivious
                        bool ascendingHalf = ((i & k) == 0);

                        int pred;
                        if (ascendingHalf) {
                            pred = static_cast<int>(cmp(arr[ij], arr[i]));
                        } else {
                            pred = static_cast<int>(cmp(arr[i], arr[ij]));
                        }

                        // 不再 if (pred) 分支，统一走 oblivious swap
                        obl_swapNode(arr[i], arr[ij], pred);
                    }
                }
            }
        }
    }

    template<typename Compare>
    static inline void imperativeXorBitonicMerge_Parallel(std::vector<Node> &arr,
                                                std::uint32_t low,
                                                std::uint32_t len,
                                                Compare cmp)
    {
        if (len < 2) return;

        // size = 最小的 2^k >= len
        std::uint32_t size = 1;
        while (size < len) size <<= 1;

        for (std::uint32_t step = (size >> 1); step > 0; step >>= 1) {

    #pragma omp parallel for schedule(static)

            for (int t = 0; t < static_cast<int>(len); ++t) {
                std::uint32_t iRel = static_cast<std::uint32_t>(t);
                std::uint32_t jRel = iRel ^ step;

                if (jRel > iRel && jRel < len) {
                    std::uint32_t i = low + iRel;
                    std::uint32_t j = low + jRel;

                    int pred = static_cast<int>(cmp(arr[j], arr[i]));
                    obl_swapNode(arr[i], arr[j], pred);
                }
            }
        }
    }

    template<typename Compare>
    static inline void xorBitonicSort_Parallel(std::vector<Node> &arr,
                                    std::uint32_t low,
                                    std::uint32_t len,
                                    Compare cmp)
    {
        if (len <= 1) return;

        std::uint32_t m = greatestPowerOfTwoLessOrEqual(len);

        if (m == len) {
            // 整段正好是 2 的幂：拷到 tmp 里做 power-of-two sort
            std::vector<Node> tmp(len);

            for (std::uint32_t i = 0; i < len; ++i) {
                Node::obl_copy_node(1, &arr[low + i], &tmp[i]);
            }

            imperativeXorBitonicSort_Parallel(tmp, cmp);

            for (std::uint32_t i = 0; i < len; ++i) {
                Node::obl_copy_node(1, &tmp[i], &arr[low + i]);
            }
        } else {
            // 左边：长度 m（2 的幂） -> 先升序，再 reverse 成降序
            std::vector<Node> tmp(m);
            for (std::uint32_t i = 0; i < m; ++i) {
                Node::obl_copy_node(1, &arr[low + i], &tmp[i]);
            }

            imperativeXorBitonicSort_Parallel(tmp, cmp);
            std::reverse(tmp.begin(), tmp.end());

            for (std::uint32_t i = 0; i < m; ++i) {
                Node::obl_copy_node(1, &tmp[i], &arr[low + i]);
            }

            // 右边：递归升序
            xorBitonicSort_Parallel(arr, low + m, len - m, cmp);

            // 整段 [low, low+len) 现在是 bitonic 序列：左降右升 -> merge 成整体升序
            imperativeXorBitonicMerge_Parallel(arr, low, len, cmp);
        }
    }

    template<typename Compare>
    void xorBitonicSort_Parallel(std::vector<Node> &arr, Compare cmp)
    {
        if (arr.size() <= 1) return;
        xorBitonicSort_Parallel(arr, 0, static_cast<uint32_t>(arr.size()), cmp);
    }

    template<typename Compare>
    void xorBitonicSort(std::vector<Node> &arr, Compare cmp)
    {
        if (arr.size() <= 1) return;
        xorBitonicSort(arr, 0, static_cast<uint32_t>(arr.size()), cmp);
    }

};

