#include "Oht.h"

#include <ctime>
#include <iostream>

Oht::Oht(size_t b, size_t z){
    kB_ = b;
    kZ_ = z;
}

void Oht::prepare(RequestBuc &&elem) {

    if (!has_prototype_) {
        prototype_      = elem;   // 深拷贝一份，里面的 Node 数组、尺寸都跟真实的一样
        has_prototype_  = true;
        // 注意：不要改 prototype_.tag，让它保持“正常请求”的 tag 形态，
        // 下面造 filler 的时候会手动设置 tag。
    }

    bins1_.push_back(std::move(elem));

}

void Oht::preparecopy(RequestBuc &&elem) {

    copybin1.push_back(std::move(elem));

}

void Oht::build() {
    build_pass(bins1_, &bins2_, 0);
    for (auto &elem : bins2_) {
        elem.tag = unset(elem.tag, kTagExcess);
    }
    build_pass(bins2_, nullptr, kB_ * kZ_);
}

void Oht::paddingnodes(int dim, NodeType type, int layerID,   int batchsize){
    for (auto &elem : bins1_){
        for (int i = 0; i < BUCKETSIZE; ++i){
            if (elem.arr[i] != nullptr){
                
            }else{
                elem.arr[i] = new Node(dim, type, layerID, batchsize);
                elem.arr[i]->isDummy = 1;
            }
            
        }
    }
    for (auto &elem : bins2_){
        for (int i = 0; i < BUCKETSIZE; ++i){
            if (elem.arr[i] != nullptr){
            }else{
                elem.arr[i] = new Node(dim, type, layerID, batchsize);
                elem.arr[i]->isDummy = 1;
            }
        }
    }
}

void Oht::build_pass(std::vector<RequestBuc> &bins, std::vector<RequestBuc> *overflow_pile, size_t oht_pos_base) {
    for (auto &elem : bins) {
        elem.bin_id = std::hash<int>{}(elem._bucID) % kB_;
    }

    // 要求在 build 之前至少 prepare 过一个真实 RequestBuc
    if (!has_prototype_) {
        std::cerr << "Oht::build_pass called without prototype RequestBuc!" << std::endl;
        std::abort();
    }

    printf("oht build, start add fillers\n");
    std::time_t now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

     // 添加 fillers：每个 bin 填充 kZ_ 个
    for (size_t i = 0; i < kB_; ++i) {
        for (size_t j = 0; j < kZ_; ++j) {
            // 用 prototype_ 拷贝构造一个 filler，
            // 这样内部 Node 数组 / 维度 和真实 RequestBuc 完全一致
            RequestBuc filler(prototype_);

            filler.tag        = kTagFiller;
            filler.bin_id     = static_cast<int>(i);
            filler._bucID     = 0;
            filler._inputRank = 100;  // 只做占位，不会被命中

            bins.push_back(std::move(filler));
        }
    }

    // 1st osort to bins with fillers in the back of the bin
    const auto cmp1 = [this](const RequestBuc &a, const RequestBuc &b) {
        int a_id = ((int(a.bin_id) * 1000) + get(a.tag, kTagFiller));
        int b_id = ((int(b.bin_id) * 1000) + get(b.tag, kTagFiller));
       return obl::OblLt(a_id, b_id);
    };

    printf("oht build, start sort 1\n");
    now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

    Requester::o_sort_ReqBuc_Parallel(bins, 0, bins.size(), cmp1);

    printf("oht build, end sort 1\n");
    now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

    printf("oht build, start assign excess\n");
    now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

    // Assign excess
    int bin_elem_num = 0;
    int last_bin_id = 0;

    for (auto &elem : bins) {
        bool same_bin = obl::OblEq((unsigned)last_bin_id,
                                (unsigned)elem.bin_id);

        size_t next_cnt = bin_elem_num + 1;
        bin_elem_num = obl::OblChoose(same_bin,
                                    next_cnt,
                                    size_t(1));

        last_bin_id = elem.bin_id;

        bool is_excess = obl::OblGt((unsigned)bin_elem_num,
                                    (unsigned)kZ_);

        unsigned tag_if_excess =
            (unsigned)elem.tag | (unsigned)kTagExcess;

        elem.tag = (decltype(elem.tag))
                obl::OblChoose(is_excess,
                                tag_if_excess,
                                (unsigned)elem.tag);
    }

    // 2nd osort to bins with the equal size `z` and a overflow pile.
    // In the overflow pile, `(k, v)`/dummy are in front of fillers.

    const auto cmp2 = +[](const RequestBuc &a, const RequestBuc &b) {
        // 主键：Excess bit（0=非 excess，1=excess）
        unsigned a_ex = get(a.tag, kTagExcess);
        unsigned b_ex = get(b.tag, kTagExcess);

        // 次键：bin_id
        unsigned a_bin = static_cast<unsigned>(a.bin_id);
        unsigned b_bin = static_cast<unsigned>(b.bin_id);

        // 三键：用 Dummy / Filler 组合一个稳定的 tie-breaker
        uint32_t a_id2 =
            (static_cast<uint32_t>(get(a.tag, kTagDummy)) << 8) |
            static_cast<uint32_t>(get(a.tag, kTagFiller));
        uint32_t b_id2 =
            (static_cast<uint32_t>(get(b.tag, kTagDummy)) << 8) |
            static_cast<uint32_t>(get(b.tag, kTagFiller));

        // 用 obl.h 做三层字典序比较（都不再用 if）

        bool ex_lt = obl::OblGt(b_ex, a_ex);
        bool ex_eq = obl::OblEq(a_ex, b_ex);

        bool bin_lt = obl::OblGt(b_bin, a_bin);
        bool bin_eq = obl::OblEq(a_bin, b_bin);

        bool id2_lt = obl::OblGt(b_id2, a_id2);

        unsigned res = 0u;

        // 第一层：Excess 优先级
        res = obl::OblChoose(ex_lt, 1u, res);

        // 第二层：Excess 相等时按 bin_id
        bool cond2 = ex_eq & bin_lt;          // 注意用按位 &，避免短路分支
        res = obl::OblChoose(cond2, 1u, res);

        // 第三层：Excess 和 bin_id 都相等时，按 Dummy/Filler 组合
        bool cond3 = ex_eq & bin_eq & id2_lt;
        res = obl::OblChoose(cond3, 1u, res);

        return res != 0u;
    };

    printf("oht build, start sort 2\n");
    now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

    Requester::o_sort_ReqBuc_Parallel(bins, 0, bins.size(), cmp2);

    printf("oht build, end sort 2\n");
    now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

    // O3 OHT Layout Reuse: record each entry's final position in `bins` so the
    // BP-time write-back can skip a full rebuild and use a single oblivious sort
    // by oht_pos to restore this layout. oht_pos_base offsets tier 2 by kB_*kZ_
    // so the two tiers' oht_pos ranges are disjoint (see sort_tiers_merged).
    // Excess entries (positions kB_*kZ_..end) get dropped by the resize below,
    // so their oht_pos values are irrelevant.
    for (size_t i = 0; i < bins.size(); ++i) {
        bins[i].oht_pos = static_cast<int>(oht_pos_base + i);
    }

    // Separate bins and the overflow pile
    if (overflow_pile != nullptr) {
        overflow_pile->reserve(bins.size() - kB_ * kZ_);
        overflow_pile->insert(overflow_pile->end(), bins.begin() + kB_ * kZ_, bins.end());
    }
    bins.resize(kB_ * kZ_);
    bins.shrink_to_fit();

    printf("oht build END\n");
    now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

}

bool Oht::lookup(RequestBuc &query) {
    int bucID = query._bucID;
    size_t bins1_bin_id = std::hash<int>{}(bucID) % kB_;
    size_t bins2_bin_id = std::hash<int>{}(bucID) % kB_;

    const auto bins1_bin_begin = bins1_.begin() + bins1_bin_id * kZ_;
    const auto bins1_bin_end = bins1_bin_begin + kZ_;
    const auto bins1_found = lookup_bin(bins1_bin_begin, bins1_bin_end, query);

    const auto bins2_bin_begin = bins2_.begin() + bins2_bin_id * kZ_;
    const auto bins2_bin_end = bins2_bin_begin + kZ_;
    const auto bins2_found = lookup_bin(bins2_bin_begin, bins2_bin_end, query);

    const auto found = bins1_found | bins2_found;

    return found; // return found or not, and the value should be already copied to query
}

bool Oht::lookup_bin(typename std::vector<RequestBuc>::iterator bin_begin, typename std::vector<RequestBuc>::iterator bin_end, RequestBuc &query) {

    bool found = false;
    for (auto elem = bin_begin; elem != bin_end; ++elem) {

        bool id_eq = (elem->_bucID == query._bucID);
        bool rank_eq = (elem->_inputRank == query._inputRank);

        bool dummy = (elem->tag & kTagDummy) > 0;
        bool filler = (elem->tag & kTagFiller) > 0;

        bool found_here = id_eq && rank_eq && !dummy && !filler;

        RequestBuc::obl_copy_reqbuc(found_here, &(*elem), &query);

        found = found || found_here;

    }
    return found;
}

bool Oht::lookupLSH_write(Bucket &query){
    int bucID = query._bucID;
    int bins1_bin_id = std::hash<int>{}(bucID) % kB_;

    int bins2_bin_id = std::hash<int>{}(bucID) % kB_;

    const auto bins1_bin_begin = bins1_.begin() + bins1_bin_id * kZ_;
    const auto bins1_bin_end = bins1_bin_begin + kZ_;
    const auto bins1_found = lookup_bin_LSH_write(bins1_bin_begin, bins1_bin_end, query);

    const auto bins2_bin_begin = bins2_.begin() + bins2_bin_id * kZ_;
    const auto bins2_bin_end = bins2_bin_begin + kZ_;
    const auto bins2_found = lookup_bin_LSH_write(bins2_bin_begin, bins2_bin_end, query);

    const auto found = bins1_found | bins2_found;

    return found; // return found or not, and the value should be already copied to query
}

bool Oht::lookup_bin_LSH_write(typename std::vector<RequestBuc>::iterator bin_begin,
                               std::vector<RequestBuc>::iterator bin_end,
                               Bucket &query)
{
    bool found_any = false;

    for (auto it = bin_begin; it != bin_end; ++it) {
        const bool id_eq     = (it->_bucID == query._bucID);
        const bool is_dummy  = (it->tag & kTagDummy);
        const bool is_filler = (it->tag & kTagFiller);
        // Bitwise & / | (no short-circuit branches). Bool domain so semantically
        // identical to && / ||.
        const bool hit       = id_eq & !(is_dummy | is_filler);

        // 命中则把 Bucket 写进对应 RequestBuc；未命中则什么都不改
        RequestBuc::obl_copy_buc_to_reqbuc(hit, &query, &(*it));
    }
    return found_any;
}

bool Oht::lookupLSH_read(Bucket &query) {
    int bucID = query._bucID;
    int bins1_bin_id = std::hash<int>{}(bucID) % kB_;
    int bins2_bin_id = std::hash<int>{}(bucID) % kB_;

    auto bins1_bin_begin = bins1_.begin() + bins1_bin_id * kZ_;
    auto bins1_bin_end   = bins1_bin_begin + kZ_;
    bool bins1_found = lookup_bin_LSH_read(bins1_bin_begin, bins1_bin_end, query);

    auto bins2_bin_begin = bins2_.begin() + bins2_bin_id * kZ_;
    auto bins2_bin_end   = bins2_bin_begin + kZ_;
    bool bins2_found = lookup_bin_LSH_read(bins2_bin_begin, bins2_bin_end, query);

    bool found = bins1_found | bins2_found;
    return found;
}

bool Oht::lookup_bin_LSH_read(std::vector<RequestBuc>::iterator bin_begin,
                              std::vector<RequestBuc>::iterator bin_end,
                              Bucket &query)
{
    bool found_any = false;

    for (auto it = bin_begin; it != bin_end; ++it) {
        const bool id_eq     = (it->_bucID == query._bucID);
        const bool is_dummy  = (it->tag & kTagDummy);
        const bool is_filler = (it->tag & kTagFiller);
        // Bitwise & / | (no short-circuit branches). Bool domain so semantically
        // identical to && / ||.
        const bool hit       = id_eq & !(is_dummy | is_filler);

        // 命中则把 RequestBuc 写回 Bucket；未命中则不改 query
        RequestBuc::obl_copy_reqbuc_to_buc(hit, &(*it), &query);
    }
    return found_any;
}

