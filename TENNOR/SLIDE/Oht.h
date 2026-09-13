



#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "Requester.h"
#include "obl.h"








class Oht {
public:
    std::vector<RequestBuc> bins1_{};
    std::vector<RequestBuc> bins2_{};
    std::vector<RequestBuc> copybin1{};
    size_t kB_;
    size_t kZ_;

    bool       has_prototype_;
    RequestBuc prototype_;

    Oht(size_t b, size_t z);

    void clear() {
        bins1_.clear();
        bins2_.clear();
        copybin1.clear();
        has_prototype_ = false;
    }

    void prepare(RequestBuc &&elem);

    void preparecopy(RequestBuc &&elem);

    void build();
    bool lookup(RequestBuc &query);

    bool lookupLSH_write(Bucket &query);

    bool lookupLSH_read(Bucket &query);




    static const uint32_t kTagFiller = 1 << 1;
    static const uint32_t kTagExcess = 1 << 2;
    static const uint32_t kTagDummy = 1 << 0;

    inline uint32_t init() {
        return 0;
    }

    inline uint32_t set(uint32_t x, uint32_t tag) {
        return x | tag;
    }

    inline uint32_t unset(uint32_t x, uint32_t tag) {
        return x & ~tag;
    }

    static inline bool check(uint32_t x, uint32_t tag) {
        return (x & tag) > 0;
    }

    static inline uint8_t get(uint32_t x, uint32_t tag) {
        return check(x, tag) ? 1 : 0;
    }

    void paddingnodes(int dim, NodeType type, int layerID,   int batchsize);



private:
    void build_pass(std::vector<RequestBuc> &bins, std::vector<RequestBuc> *overflow_pile);

    bool lookup_bin(typename std::vector<RequestBuc>::iterator bin_begin, typename std::vector<RequestBuc>::iterator bin_end, RequestBuc &query);

    bool lookup_bin_LSH_write(typename std::vector<RequestBuc>::iterator bin_begin, typename std::vector<RequestBuc>::iterator bin_end, Bucket &query);

    bool lookup_bin_LSH_read(typename std::vector<RequestBuc>::iterator bin_begin, typename std::vector<RequestBuc>::iterator bin_end, Bucket &query);
};



