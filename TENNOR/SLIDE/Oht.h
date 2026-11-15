#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "Requester.h"
#include "obl.h"

// OHT should be contianed by each Layer
// the size of OHT -- num of bins b, and size of bin z should be defined in the Config.h

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

    // Merge bins1_ and bins2_, run ONE oblivious sort with `cmp`, split back
    // (bins1_ takes the first bins1_.size() entries). This is how the two-tier
    // OHT paper sorts: both tiers as a single unit, so the access trace does not
    // depend on whether anything overflowed into tier 2. Entries are moved, not
    // copied (RequestBuc move ctor / move= transfer the Node** arr pointer); the
    // deep copies happen only inside the sort's obl_swap_reqbuc, same as they
    // would for a single-tier sort of this total length.
    template <typename Compare>
    void sort_tiers_merged(Compare cmp) {
        const size_t n1 = bins1_.size();
        const size_t n2 = bins2_.size();
        std::vector<RequestBuc> merged;
        merged.reserve(n1 + n2);
        for (auto &e : bins1_) merged.push_back(std::move(e));
        for (auto &e : bins2_) merged.push_back(std::move(e));
        Requester::o_sort_ReqBuc_Parallel(merged, 0, static_cast<uint32_t>(n1 + n2), cmp);
        for (size_t i = 0; i < n1; ++i) bins1_[i] = std::move(merged[i]);
        for (size_t i = 0; i < n2; ++i) bins2_[i] = std::move(merged[n1 + i]);
    }

private:
    // oht_pos_base is added to every recorded oht_pos: bins1_ gets [0, kB_*kZ_),
    // bins2_ gets [kB_*kZ_, 2*kB_*kZ_). Keeps the two tiers' oht_pos ranges
    // disjoint so a merged sort by oht_pos has no cross-tier ties and splits back
    // to the exact FF layout.
    void build_pass(std::vector<RequestBuc> &bins, std::vector<RequestBuc> *overflow_pile, size_t oht_pos_base);
    bool lookup_bin(typename std::vector<RequestBuc>::iterator bin_begin, typename std::vector<RequestBuc>::iterator bin_end, RequestBuc &query);

    bool lookup_bin_LSH_write(typename std::vector<RequestBuc>::iterator bin_begin, typename std::vector<RequestBuc>::iterator bin_end, Bucket &query);

    bool lookup_bin_LSH_read(typename std::vector<RequestBuc>::iterator bin_begin, typename std::vector<RequestBuc>::iterator bin_end, Bucket &query);
};

