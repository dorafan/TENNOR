#pragma once

#include <utility>                 

#include <vector>                       
#include <cmath>                                      
#include <algorithm>                             

#include "Bucket.h"

#include "omp.h"

class RequestBuc : public Bucket {
public:
    int _inputRank = 0;

    // used for OHT to identify types
    int tag = 0;

    int bin_id = 0;

    // O3 OHT Layout Reuse: final position in bins1_ after FF build_pass sort2.
    // Set in Oht::build_pass; preserved through obl_copy_reqbuc (RequestBucOwnScalarBlock
    // covers it) and §9 merge (which only touches arr[i].t/_tbias and tag).
    int oht_pos = 0;

    RequestBuc(int dim, NodeType type, int layerID,  int batchsize);

    int getRequestTag();

    void setRequestTag(int tag);

    static void obl_copy_reqbuc(bool pred, RequestBuc* source, RequestBuc* dest);

    static void obl_copy_buc_to_reqbuc(bool pred, const Bucket* source, RequestBuc* dest);

     // RequestBuc -> Bucket 的 oblivious copy
    static void obl_copy_reqbuc_to_buc(bool pred,
                                       const RequestBuc* source,
                                       Bucket* dest);

    void mergeWith(const RequestBuc* other);

    // Oblivious fold zeros other's _t/_tbias when pred=true.
    void obl_mergeWith(RequestBuc *other, bool pred);

    RequestBuc() : Bucket(), _inputRank(0), tag(0), bin_id(0), oht_pos(0) {}

    RequestBuc(const RequestBuc& other)
            : Bucket(other), _inputRank(other._inputRank), tag(other.tag),
              bin_id(other.bin_id), oht_pos(other.oht_pos)
    {

    }

    RequestBuc& operator=(const RequestBuc& other)
    {

        if (this == &other) return *this;
        Bucket::operator=(other);
        _inputRank = other._inputRank;
        tag = other.tag;
        bin_id = other.bin_id;
        oht_pos = other.oht_pos;
        return *this;
    }

    void copyFromBucket(const Bucket& bucket) {
        Bucket::operator=(bucket);
    }

    void copyToBucket(Bucket &dest) const {
        static_cast<Bucket&>(dest) = static_cast<const Bucket&>(*this);

    }

    RequestBuc(RequestBuc&& other) noexcept
            : Bucket(std::move(other)), _inputRank(other._inputRank), tag(other.tag),
              bin_id(other.bin_id), oht_pos(other.oht_pos)
    {

        other._inputRank = 0;
        other.tag = 0;
        other.bin_id = 0;
        other.oht_pos = 0;
    }

    RequestBuc& operator=(RequestBuc&& other) noexcept
    {

        if (this == &other) return *this;
        Bucket::operator=(std::move(other));
        _inputRank = other._inputRank;
        tag = other.tag;
        bin_id = other.bin_id;
        oht_pos = other.oht_pos;
        other._inputRank = 0;
        other.tag = 0;
        other.bin_id = 0;
        other.oht_pos = 0;
        return *this;
    }

    ~RequestBuc () override
    {
    }

};

class Requester {
private:

public:
    int _currentReqIndex = 0;
    int _totalReqs = 0;
    RequestBuc **_reqTable = nullptr;

    Requester(int totalReqs);

    ~Requester();

    void generateRequestBucs(int* requestIDs, int numRequests, int inputID, int dim, NodeType type, int layerID,  int batchsize);

    void sortByID();

    void sortByInputRank();

    void obl_mergeSameBucID(int& totalReqMerge);

    template <class Compare>
    static void o_sort_ReqBuc_New(std::vector<RequestBuc>& bins,
                              size_t begin,
                              size_t end,
                              Compare cmp);

    void mergeSameBucID(int& totalReqMerge);

    void mergeRequestBucs(int start, int end);

    void removeNullPointers(int& totalReqMerge);

    inline bool lessReqBuc(const RequestBuc &a, const RequestBuc &b) {
        return a._inputRank < b._inputRank;
    }

    inline static void swapRequestBuc(RequestBuc& lhs, RequestBuc& rhs) noexcept {
        if (&lhs == &rhs) {
            return;
        }

        RequestBuc temp(std::move(lhs));
        lhs = std::move(rhs);
        rhs = std::move(temp);
    }

    inline static uint32_t log2_ceil(uint32_t x) {
        // For x>0, an integer ceil of log2(x).
        // Example: if x=5..8 => returns 3; if x=9..16 => returns 4
        if (x <= 1) return 0;
        double logv = std::log2((double)x);
        return (uint32_t) std::ceil(logv);
    }

// Greatest power of two <= x
    inline static uint32_t greatest_power_of_two_less_than(uint32_t x) {
        // If x is already a power of two, returns x.
        // Otherwise returns the next lower power of two.
        // Example: if x=10 => returns 8; if x=16 => returns 16
        if (x < 1) return 0;
        // Shift x right until only one '1' remains
        uint32_t p = 1;
        while (p <= x) {
            p <<= 1;
        }
        return p >> 1;
    }

    /**
 * @brief Bitonic sort on a subarray [0..n) of `arr`, assuming n is a power of two.
 * @tparam Compare A comparator callable: bool cmp(const RequestBuc&, const RequestBuc&).
 */
    template<typename Compare>
    static void imperative_o_sort_ReqBuc(std::vector<RequestBuc> &arr, Compare cmp, RequestBuc &scratch)
    {
        size_t n = arr.size();
        // For a correct bitonic sort, n should be a power of two.
        // If it's not, see the o_sort_ReqBuc wrapper.

        for (size_t k = 2; k <= n; k <<= 1) {
            for (size_t j = k >> 1; j > 0; j >>= 1) {
                for (size_t i = 0; i < n; i++) {
                    size_t ij = i ^ j;
                    if (ij > i) {
                        if ((i & k) == 0) {
                            bool pred = cmp(arr[ij], arr[i]);
                            obl_swap_reqbuc(pred, &arr[i], &arr[ij], scratch);
                        } else {
                            bool pred = cmp(arr[i], arr[ij]);
                            obl_swap_reqbuc(pred, &arr[i], &arr[ij], scratch);
                        }
                    }
                }
            }
        }
    }

/**
 * @brief Merge routine for bitonic approach (sub-blocks in ascending order).
 * @tparam Compare A comparator callable: bool cmp(const RequestBuc&, const RequestBuc&).
 */
    template<typename Compare>
    static void imperative_o_merge_ReqBuc(std::vector<RequestBuc> &arr,
                                          uint32_t low, uint32_t len,
                                          Compare cmp,
                                          RequestBuc &scratch)
    {
        uint32_t l = log2_ceil(len);
        uint32_t n = 1 << l;

        for (uint32_t i = 0; i < l; i++) {
            uint32_t step = n >> i;
            for (uint32_t j = 0; j < n; j += step) {
                for (uint32_t k = 0; k < step/2; k++) {
                    uint32_t i1 = low + j + k;
                    uint32_t i2 = i1 + step/2;
                    if (i2 >= low + len) {
                        break;
                    }
                    bool pred = cmp(arr[i2], arr[i1]);
                    obl_swap_reqbuc(pred, &arr[i1], &arr[i2], scratch);
                }
            }
        }
    }

/**
 * @brief Oblivious sort wrapper that handles non-power-of-two sizes
 *        by splitting and merging into a bitonic sequence.
 * @tparam Compare The comparator type.
 */
    template<typename Compare>
    static void o_sort_ReqBuc(std::vector<RequestBuc> &arr,
                       uint32_t low, uint32_t len,
                       Compare cmp)
    {
        if (len <= 1) {
            return;
        }

        // 用子区间第一个元素构造一个 scratch，保证 arr/Node 都分配好了
        RequestBuc scratch = arr[low];
        
        if (len > 1) {
            uint32_t m = greatest_power_of_two_less_than(len);
            if (m * 2 == len) {
                std::vector<RequestBuc> tmp(arr.begin() + low, arr.begin() + low + len);
                imperative_o_sort_ReqBuc(tmp, cmp, scratch);
                for (uint32_t i = 0; i < len; i++) {
                    arr[low + i] = tmp[i];
                }
            } else {
                // Not a perfect power of two
                // 1) Sort [low..low+m) in ascending, then reverse => effectively descending
                std::vector<RequestBuc> tmp1(arr.begin() + low, arr.begin() + low + m);
                imperative_o_sort_ReqBuc(tmp1, cmp, scratch);
                std::reverse(tmp1.begin(), tmp1.end());
                for (uint32_t i = 0; i < m; i++) {
                    arr[low + i] = tmp1[i];
                }

                o_sort_ReqBuc(arr, low + m, len - m, cmp);

                imperative_o_merge_ReqBuc(arr, low, len, cmp, scratch);
            }
        }
    }

    /**
 * @brief Sorts a sub-array [arr..arr+n) using bitonic logic,
 *        assuming n is a power of two, and always ascending by comparator `cmp`.
 * @tparam Compare A comparator: bool cmp(const RequestBuc*, const RequestBuc*)
 */
    template<typename Compare>
    void imperative_o_sort_ReqBucPtr(RequestBuc** arr, size_t n, Compare cmp, RequestBuc& scratch)
    {

        for (size_t k = 2; k <= n; k <<= 1) {
            for (size_t j = k >> 1; j > 0; j >>= 1) {
                for (size_t i = 0; i < n; i++) {
                    size_t ij = i ^ j;
                    if (ij > i) {

                        bool up = ((i & k) == 0);
                        bool pred;
                        if (up) {
                            pred = cmp(arr[ij], arr[i]);
                        } else {
                            pred = cmp(arr[i], arr[ij]);
                        }

                        // 不再在 pred 上分支，而是做 oblivious 深度 swap
                        obl_swap_reqbuc(pred, arr[i], arr[ij], scratch);

                    }
                }
            }
        }
    }

    template<typename Compare>
    void imperative_o_merge_ReqBucPtr(RequestBuc** arr, uint32_t low, uint32_t len, Compare cmp, RequestBuc& scratch)
    {
        // Compute the next power-of-two block that covers 'len'
        uint32_t l = 0;
        {
            uint32_t temp = len;
            while (temp > 1) {
                temp = (temp + 1) >> 1;
                l++;
            }
        }
        uint32_t n = 1 << l;  // next power of two >= len

        for (uint32_t i = 0; i < l; i++) {
            uint32_t step = n >> i;
            for (uint32_t j = 0; j < n; j += step) {
                for (uint32_t k = 0; k < step / 2; k++) {
                    uint32_t i1 = low + j + k;
                    uint32_t i2 = i1 + (step / 2);
                    if (i2 >= low + len) {
                        break;
                    }
                    bool pred = cmp(arr[i2], arr[i1]);
                    obl_swap_reqbuc(pred, arr[i1], arr[i2], scratch);
                }
            }
        }
    }

    /**
 * @brief Sort sub-array [low..low+len) in ascending order,
 *        handling non-power-of-two sizes by partial approach,
 *        building a bitonic sequence, then merging.
 * @tparam Compare bool cmp(const RequestBuc*, const RequestBuc*).
 */
    template<typename Compare>
    void o_sort_ReqBucPtr(RequestBuc** arr, uint32_t low, uint32_t len, Compare cmp, RequestBuc& scratch)
    {
        if (len > 1) {
            uint32_t m = greatest_power_of_two_less_than(len);
            if (m * 2 == len) {
                imperative_o_sort_ReqBucPtr(arr + low, len, cmp, scratch);
            } else {

                imperative_o_sort_ReqBucPtr(arr + low, m, cmp, scratch);

                for (uint32_t i = 0; i < m / 2; i++) {
                    std::swap(arr[low + i], arr[low + (m - 1 - i)]);
                }

                o_sort_ReqBucPtr(arr, low + m, len - m, cmp, scratch);

                imperative_o_merge_ReqBucPtr(arr, low, len, cmp, scratch);
            }
        }
    }

    void sortByIDBitonicPtr();

    /**
 * @brief Deep-swap: Exchange the *contents* of *lhs and *rhs,
 *        using the move constructor and move assignment operators.
 *        That is, *lhs becomes what *rhs was, and vice versa.
 */
    inline static void swapRequestBucTable(RequestBuc* lhs, RequestBuc* rhs) noexcept {
        if (lhs == rhs) {
            return;
        }

        RequestBuc temp(std::move(*lhs));
        *lhs = std::move(*rhs);
        *rhs = std::move(temp);
    }

    inline static void obl_swap_reqbuc(bool pred,
                                   RequestBuc* a,
                                   RequestBuc* b,
                                   RequestBuc& scratch)
    {
        // 这里默认 a、b、scratch 非空，且三者的 arr[i] 都已经分配好了
        //（这是 RequestBuc 构造 + 拷贝构造保证的）

        RequestBuc::obl_copy_reqbuc(pred, a, &scratch);
        RequestBuc::obl_copy_reqbuc(pred, b, a);
        RequestBuc::obl_copy_reqbuc(pred, &scratch, b);
    }

    inline static void obl_swap_ReqBuc2(bool pred,
                                       RequestBuc &a,
                                       RequestBuc &b,
                                       RequestBuc &scratch)
    {
        RequestBuc::obl_copy_reqbuc(true, &a, &scratch);

        RequestBuc::obl_copy_reqbuc(pred, &b, &a);

        RequestBuc::obl_copy_reqbuc(pred, &scratch, &b);
    }

    // 并行版 bitonic 排序：对整个 vector [0..n) 做 oblivious sort
    template<typename Compare>
    static void imperative_o_sort_ReqBuc_Parallel(std::vector<RequestBuc> &arr,
                                        Compare cmp,
                                        RequestBuc & /*scratch_unused*/)
    {
        size_t n = arr.size();
        if (n <= 1) return;

    #pragma omp parallel
        {
            // 每个线程自己的 scratch，拷贝构造保证 Node/arr 已经分配好了
            RequestBuc scratch_local = arr[0];

            for (size_t k = 2; k <= n; k <<= 1) {
                for (size_t j = k >> 1; j > 0; j >>= 1) {

    #pragma omp for schedule(static)
                    for (int i = 0; i < static_cast<int>(n); ++i) {
                        size_t iu = static_cast<size_t>(i);
                        size_t ij = iu ^ j;
                        if (ij > iu) {
                            bool up = ((iu & k) == 0);

                            bool pred;
                            if (up) {
                                pred = cmp(arr[ij], arr[iu]);
                            } else {
                                pred = cmp(arr[iu], arr[ij]);
                            }

                            // 用线程私有的 scratch_local 深拷贝 swap
                            obl_swap_reqbuc(pred, &arr[iu], &arr[ij], scratch_local);
                        }
                    }
                    // 每个 stage 结束有隐式 barrier
                }
            }
        }
    }

    // 并行版 merge：对 arr[low .. low+len) 做 bitonic merge
    template<typename Compare>
    static void imperative_o_merge_ReqBuc_Parallel(std::vector<RequestBuc> &arr,
                                                uint32_t low, uint32_t len,
                                                Compare cmp,
                                                RequestBuc &scratch)
    {
        if (len <= 1) return;

        uint32_t l = log2_ceil(len);
        uint32_t n = 1u << l;     // n 是 2 的幂，>= len

    #pragma omp parallel firstprivate(scratch)
        {
            // 每个线程有自己的 scratch 副本
            RequestBuc &scratch_local = scratch;

            for (uint32_t level = 0; level < l; ++level) {
                uint32_t step  = n >> level;
                uint32_t half  = step >> 1;
                uint32_t pairs = n >> 1;

                // 把所有 compare–swap pair 用一个 t 索引拍平，便于并行
    #pragma omp for schedule(static)
                for (int t = 0; t < static_cast<int>(pairs); ++t) {
                    uint32_t block  = static_cast<uint32_t>(t) / half;
                    uint32_t offset = static_cast<uint32_t>(t) % half;

                    uint32_t j  = block * step;
                    uint32_t i1 = low + j + offset;
                    uint32_t i2 = i1 + half;

                    if (i2 >= low + len) {
                        continue;
                    }

                    bool pred = cmp(arr[i2], arr[i1]);
                    // 用线程私有 scratch 做 oblivious 深拷贝 swap
                    obl_swap_reqbuc(pred, &arr[i1], &arr[i2], scratch_local);
                }
                // 每一层结束有隐式 barrier
            }
        }
    }

    template<typename Compare>
    static void o_sort_ReqBuc_Parallel(std::vector<RequestBuc> &arr,
                            uint32_t low, uint32_t len,
                            Compare cmp)
    {
        if (len <= 1) {
            return;
        }

        // 用子区间第一个元素构造一个 scratch，保证 Node/arr 初始化完整
        RequestBuc scratch = arr[low];

        uint32_t m = greatest_power_of_two_less_than(len);
        if (m * 2 == len) {
            // 情况 1：len 本身是 2 的幂（m*2 == len），直接对这个块做 bitonic sort
            std::vector<RequestBuc> tmp(arr.begin() + low, arr.begin() + low + len);

            imperative_o_sort_ReqBuc_Parallel(tmp, cmp, scratch);

            for (uint32_t i = 0; i < len; ++i) {
                arr[low + i] = tmp[i];
            }
        } else {
            // 情况 2：len 不是 2 的幂
            // 1) 先对 [low .. low+m) 做升序 bitonic sort，然后 reverse 成降序
            std::vector<RequestBuc> tmp1(arr.begin() + low, arr.begin() + low + m);
            imperative_o_sort_ReqBuc_Parallel(tmp1, cmp, scratch);
            std::reverse(tmp1.begin(), tmp1.end());
            for (uint32_t i = 0; i < m; ++i) {
                arr[low + i] = tmp1[i];
            }

            o_sort_ReqBuc_Parallel(arr, low + m, len - m, cmp);

            imperative_o_merge_ReqBuc_Parallel(arr, low, len, cmp, scratch);
        }
    }

    // 并行版 bitonic 排序：对指针数组 arr[0..n) 做 oblivious sort
    template<typename Compare>
    static void imperative_o_sort_ReqBucPtr_Parallel(RequestBuc** arr,
                                                    uint32_t n,
                                                    Compare cmp)
    {
        if (n <= 1) return;

    #pragma omp parallel
        {
            // 每个线程自己的 scratch，拷贝构造保证内部 Node/arr 都初始化好了
            RequestBuc scratch_local = *arr[0];

            for (uint32_t k = 2; k <= n; k <<= 1) {
                for (uint32_t j = k >> 1; j > 0; j >>= 1) {

    #pragma omp for schedule(static)
                    for (int i = 0; i < static_cast<int>(n); ++i) {
                        uint32_t iu = static_cast<uint32_t>(i);
                        uint32_t ij = iu ^ j;
                        if (ij > iu) {
                            bool up = ((iu & k) == 0);

                            bool pred;
                            if (up) {
                                pred = cmp(arr[ij], arr[iu]);
                            } else {
                                pred = cmp(arr[iu], arr[ij]);
                            }

                            // 用线程私有的 scratch_local 做 oblivious swap
                            obl_swap_reqbuc(pred, arr[iu], arr[ij], scratch_local);
                        }
                    }
                    // 每个 stage 末尾有隐式 barrier，确保这一层全部完成
                }
            }
        }
}

// 并行版 pointer merge：对 arr[low .. low+len) 做 bitonic merge
    template<typename Compare>
    static void imperative_o_merge_ReqBucPtr_Parallel(RequestBuc** arr,
                                                    uint32_t low, uint32_t len,
                                                    Compare cmp)
    {
        if (len <= 1) return;

        uint32_t l = 0;
        {
            uint32_t temp = len;
            while (temp > 1) {
                temp = (temp + 1) >> 1;
                l++;
            }
        }
        uint32_t n = 1u << l;  // >= len 且是 2 的幂

    #pragma omp parallel
        {
            // 每个线程自己的 scratch，结构和 RequestBuc 一样
            RequestBuc scratch_local = *arr[low];  // 假设 low 位置一定有合法 bucket

            for (uint32_t level = 0; level < l; ++level) {
                uint32_t step  = n >> level;
                uint32_t half  = step >> 1;
                uint32_t pairs = n >> 1;

    #pragma omp for schedule(static)
                for (int t = 0; t < static_cast<int>(pairs); ++t) {
                    uint32_t block  = static_cast<uint32_t>(t) / half;
                    uint32_t offset = static_cast<uint32_t>(t) % half;

                    uint32_t j  = block * step;
                    uint32_t i1 = low + j + offset;
                    uint32_t i2 = i1 + half;

                    if (i2 >= low + len) {
                        continue;
                    }

                    bool pred = cmp(arr[i2], arr[i1]);
                    obl_swap_reqbuc(pred, arr[i1], arr[i2], scratch_local);
                }
            }
        }
    }

// 并行版封装：处理非 2 的幂长度，内部调用并行 bitonic 内核
    template<typename Compare>
    static void o_sort_ReqBucPtr_Parallel(RequestBuc** arr,
                                        uint32_t low, uint32_t len,
                                        Compare cmp)
    {
        if (len <= 1) {
            return;
        }

        uint32_t m = greatest_power_of_two_less_than(len);

        if (m * 2 == len) {
            imperative_o_sort_ReqBucPtr_Parallel(arr + low, len, cmp);
        } else {

            // 1) 先对 [low .. low+m) 做升序 bitonic sort
            imperative_o_sort_ReqBucPtr_Parallel(arr + low, m, cmp);

            for (uint32_t i = 0; i < m / 2; i++) {
                std::swap(arr[low + i], arr[low + (m - 1 - i)]);
            }

            o_sort_ReqBucPtr_Parallel(arr, low + m, len - m, cmp);

            imperative_o_merge_ReqBucPtr_Parallel(arr, low, len, cmp);
        }
    }

};

template <class Compare>
void Requester::o_sort_ReqBuc_New(std::vector<RequestBuc>& bins,
                            size_t begin,
                            size_t end,
                            Compare cmp)
{
    size_t n = (end > begin) ? (end - begin) : 0;
    if (n <= 1) return;

    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) {
        idx[i] = i;
    }

    auto idx_cmp = [&](size_t ia, size_t ib) -> bool {
        return cmp(bins[begin + ia], bins[begin + ib]);
    };

    // 2) 在 idx 上做 bitonic sort（访问模式只依赖 i/j/k，这些都公开）
    //    注意：这里唯一跟秘密相关的是 idx_cmp 的比较结果，但我们不用它做分支，
    //          只用它喂给 OblChoose 做无分支交换。
    for (size_t k = 2; k <= n; k <<= 1) {
        for (size_t j = k >> 1; j > 0; j >>= 1) {
            for (size_t i = 0; i < n; ++i) {
                size_t l = i ^ j;
                if (l <= i) continue;   // 这条分支只依赖 i/j，是公开的

                bool dir  = ((i & k) == 0);           // 前半段升序，后半段降序（公开）
                bool less = idx_cmp(idx[i], idx[l]);

                // 升序段：a> b 时交换；降序段：a< b 时交换
                bool swapAsc  = !less;
                bool swapDesc =  less;
                bool doSwap   = dir ? swapAsc : swapDesc;  // dir 公开，这个条件是“公开选择”

                size_t a = idx[i];
                size_t b = idx[l];

                // 用 OblChoose 做无分支交换
                idx[i] = obl::OblChoose(doSwap, b, a);
                idx[l] = obl::OblChoose(doSwap, a, b);
            }
        }
    }

    // 3) 先把原来的 bins[begin..end) 复制一份到 orig 里（一次 O(n) 深拷贝）
    std::vector<RequestBuc> orig(n);
    for (size_t t = 0; t < n; ++t) {
        orig[t] = bins[begin + t];
    }

    // 4) 完全 oblivious 的重排：
    //    对于每个目标位置 dest，我们扫一遍所有 src，
    //    pred = (idx[dest] == src)，如果为 1，就把 orig[src] obl_copy 到 bins[begin+dest]。
    //    外层 dest、内层 src 的循环边界是固定的，访问 orig/src 和 bins/dest
    //    的模式只依赖公开的 (dest, src)，跟秘密排序结果无关。

    // 4) 重排阶段用 OpenMP 并行 dest 维度
    #pragma omp parallel for schedule(static)
    for (long long dest = 0; dest < static_cast<long long>(n); ++dest) {
        for (size_t src = 0; src < n; ++src) {
            bool pred = obl::OblEq(
                static_cast<unsigned>(idx[dest]),
                static_cast<unsigned>(src)
            );
            RequestBuc::obl_copy_reqbuc(
                pred,
                &orig[src],
                &bins[begin + dest]
            );
        }
    }
}

