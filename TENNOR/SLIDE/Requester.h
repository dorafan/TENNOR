







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


    int tag = 0;


    int bin_id = 0;




    int oht_pos = 0;









    RequestBuc(int dim, NodeType type, int layerID,  int batchsize);





    int getRequestTag();


    void setRequestTag(int tag);






    static void obl_copy_reqbuc(bool pred, RequestBuc* source, RequestBuc* dest);

    static void obl_copy_buc_to_reqbuc(bool pred, const Bucket* source, RequestBuc* dest);


    static void obl_copy_reqbuc_to_buc(bool pred,
                                       const RequestBuc* source,
                                       Bucket* dest);

    void mergeWith(const RequestBuc* other);


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



        if (x <= 1) return 0;
        double logv = std::log2((double)x);
        return (uint32_t) std::ceil(logv);
    }


    inline static uint32_t greatest_power_of_two_less_than(uint32_t x) {



        if (x < 1) return 0;


        uint32_t p = 1;
        while (p <= x) {
            p <<= 1;
        }
        return p >> 1;
    }







    template<typename Compare>
    static void imperative_o_sort_ReqBuc(std::vector<RequestBuc> &arr, Compare cmp, RequestBuc &scratch)
    {
        size_t n = arr.size();



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






    template<typename Compare>
    static void o_sort_ReqBuc(std::vector<RequestBuc> &arr,
                       uint32_t low, uint32_t len,
                       Compare cmp)
    {
        if (len <= 1) {
            return;
        }


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


                        obl_swap_reqbuc(pred, arr[i], arr[ij], scratch);


                    }
                }
            }
        }
    }


    template<typename Compare>
    void imperative_o_merge_ReqBucPtr(RequestBuc** arr, uint32_t low, uint32_t len, Compare cmp, RequestBuc& scratch)
    {


        uint32_t l = 0;
        {

            uint32_t temp = len;
            while (temp > 1) {
                temp = (temp + 1) >> 1;
                l++;
            }
        }
        uint32_t n = 1 << l;

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








    template<typename Compare>
    static void imperative_o_sort_ReqBuc_Parallel(std::vector<RequestBuc> &arr,
                                        Compare cmp,
                                        RequestBuc & )
    {
        size_t n = arr.size();
        if (n <= 1) return;

    #pragma omp parallel
        {

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


                            obl_swap_reqbuc(pred, &arr[iu], &arr[ij], scratch_local);
                        }
                    }

                }
            }
        }
    }



    template<typename Compare>
    static void imperative_o_merge_ReqBuc_Parallel(std::vector<RequestBuc> &arr,
                                                uint32_t low, uint32_t len,
                                                Compare cmp,
                                                RequestBuc &scratch)
    {
        if (len <= 1) return;

        uint32_t l = log2_ceil(len);
        uint32_t n = 1u << l;

    #pragma omp parallel firstprivate(scratch)
        {

            RequestBuc &scratch_local = scratch;

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

                    obl_swap_reqbuc(pred, &arr[i1], &arr[i2], scratch_local);
                }

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


        RequestBuc scratch = arr[low];

        uint32_t m = greatest_power_of_two_less_than(len);
        if (m * 2 == len) {

            std::vector<RequestBuc> tmp(arr.begin() + low, arr.begin() + low + len);


            imperative_o_sort_ReqBuc_Parallel(tmp, cmp, scratch);


            for (uint32_t i = 0; i < len; ++i) {
                arr[low + i] = tmp[i];
            }
        } else {


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













    template<typename Compare>
    static void imperative_o_sort_ReqBucPtr_Parallel(RequestBuc** arr,
                                                    uint32_t n,
                                                    Compare cmp)
    {
        if (n <= 1) return;

    #pragma omp parallel
        {

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


                            obl_swap_reqbuc(pred, arr[iu], arr[ij], scratch_local);
                        }
                    }

                }
            }
        }
}




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
        uint32_t n = 1u << l;

    #pragma omp parallel
        {

            RequestBuc scratch_local = *arr[low];

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




    for (size_t k = 2; k <= n; k <<= 1) {
        for (size_t j = k >> 1; j > 0; j >>= 1) {
            for (size_t i = 0; i < n; ++i) {
                size_t l = i ^ j;
                if (l <= i) continue;

                bool dir  = ((i & k) == 0);
                bool less = idx_cmp(idx[i], idx[l]);


                bool swapAsc  = !less;
                bool swapDesc =  less;
                bool doSwap   = dir ? swapAsc : swapDesc;

                size_t a = idx[i];
                size_t b = idx[l];


                idx[i] = obl::OblChoose(doSwap, b, a);
                idx[l] = obl::OblChoose(doSwap, a, b);
            }
        }
    }


    std::vector<RequestBuc> orig(n);
    for (size_t t = 0; t < n; ++t) {
        orig[t] = bins[begin + t];
    }





















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














































