


#include "Oht.h"

#include <ctime>
#include <iostream>

































Oht::Oht(size_t b, size_t z){
    kB_ = b;
    kZ_ = z;
}


void Oht::prepare(RequestBuc &&elem) {

    if (!has_prototype_) {

        prototype_      = elem;
        has_prototype_  = true;


    }

    bins1_.push_back(std::move(elem));


}

void Oht::preparecopy(RequestBuc &&elem) {

    copybin1.push_back(std::move(elem));


}



void Oht::build() {




    build_pass(bins1_, &bins2_);
    for (auto &elem : bins2_) {
        elem.tag = unset(elem.tag, kTagExcess);
    }
    build_pass(bins2_, nullptr);
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


void Oht::build_pass(std::vector<RequestBuc> &bins, std::vector<RequestBuc> *overflow_pile) {

    for (auto &elem : bins) {

        elem.bin_id = std::hash<int>{}(elem._bucID) % kB_;
    }


    if (!has_prototype_) {
        std::cerr << "Oht::build_pass called without prototype RequestBuc!" << std::endl;
        std::abort();
    }


    printf("oht build, start add fillers\n");
    std::time_t now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;



















    for (size_t i = 0; i < kB_; ++i) {
        for (size_t j = 0; j < kZ_; ++j) {


            RequestBuc filler(prototype_);

            filler.tag        = kTagFiller;
            filler.bin_id     = static_cast<int>(i);
            filler._bucID     = 0;
            filler._inputRank = 100;

            bins.push_back(std::move(filler));
        }
    }





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

























































    const auto cmp2 = +[](const RequestBuc &a, const RequestBuc &b) {

        unsigned a_ex = get(a.tag, kTagExcess);
        unsigned b_ex = get(b.tag, kTagExcess);


        unsigned a_bin = static_cast<unsigned>(a.bin_id);
        unsigned b_bin = static_cast<unsigned>(b.bin_id);


        uint32_t a_id2 =
            (static_cast<uint32_t>(get(a.tag, kTagDummy)) << 8) |
            static_cast<uint32_t>(get(a.tag, kTagFiller));
        uint32_t b_id2 =
            (static_cast<uint32_t>(get(b.tag, kTagDummy)) << 8) |
            static_cast<uint32_t>(get(b.tag, kTagFiller));




        bool ex_lt = obl::OblGt(b_ex, a_ex);
        bool ex_eq = obl::OblEq(a_ex, b_ex);


        bool bin_lt = obl::OblGt(b_bin, a_bin);
        bool bin_eq = obl::OblEq(a_bin, b_bin);


        bool id2_lt = obl::OblGt(b_id2, a_id2);


        unsigned res = 0u;


        res = obl::OblChoose(ex_lt, 1u, res);


        bool cond2 = ex_eq & bin_lt;
        res = obl::OblChoose(cond2, 1u, res);


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





    for (size_t i = 0; i < bins.size(); ++i) {
        bins[i].oht_pos = static_cast<int>(i);
    }


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













    return found;
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



    return found;
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


        const bool hit       = id_eq & !(is_dummy | is_filler);













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


        const bool hit       = id_eq & !(is_dummy | is_filler);












        RequestBuc::obl_copy_reqbuc_to_buc(hit, &(*it), &query);

    }
    return found_any;
}





























