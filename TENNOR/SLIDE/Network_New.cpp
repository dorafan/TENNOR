#include "Network_New.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include "Config.h"
#include <fstream>
#include <omp.h>
#include <atomic>

#include <ctime>

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <ctime>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <utility>
#include <vector>
#include <cstdint>
#include <chrono>
#include <string>

#define DEBUG 1
using namespace std;

static std::atomic<int> G_THREADS{1};
static int max_threads() { return omp_get_max_threads(); }

/* 可选但强烈建议：关闭 OpenMP 的动态调度，否则运行时会擅自改线程数 */
static void init_omp() {
    omp_set_dynamic(0);  // 必须在并行区外
}

#ifndef TRACE_PER_BIN
#define TRACE_PER_BIN 1
#endif

// 统一落盘：同时写 tidy CSV（成员与计数）+ CSR 二进制（starts/bin2bucket）
static void dump_bin_layout(const std::string& stem,
                            uint32_t B,
                            const std::vector<uint32_t>& starts,
                            const std::vector<uint32_t>& bin2bucket) {
    if (starts.size() != static_cast<size_t>(B) + 1) {
        throw std::runtime_error("starts.size() != B + 1");
    }
    if (starts.back() != bin2bucket.size()) {
        throw std::runtime_error("starts.back() != bin2bucket.size()");
    }

    // 1) 每个 (bin, rank, bucket) 一行，整洁格式，适合后续 groupby/画图
    {
        std::ofstream out(stem + "_members.csv");
        out << "bin,rank,bucket\n";
        for (uint32_t b = 0; b < B; ++b) {
            uint32_t begin = starts[b], end = starts[b + 1];
            for (uint32_t r = 0; r < end - begin; ++r) {
                uint32_t bucket = bin2bucket[begin + r];
                out << b << ',' << r << ',' << bucket << '\n';
            }
        }
    }

    // 2) 每个 bin 的 bucket 数量（直方图/分布图最方便）
    {
        std::ofstream out(stem + "_counts.csv");
        out << "bin,count\n";
        for (uint32_t b = 0; b < B; ++b) {
            out << b << ',' << (starts[b + 1] - starts[b]) << '\n';
        }
    }

    // 3) CSR 二进制（快速重载/复现）：uint32_t 数组直接写
    {
        std::ofstream fa(stem + "_starts.bin", std::ios::binary);
        fa.write(reinterpret_cast<const char*>(starts.data()),
                 sizeof(uint32_t) * starts.size());

        std::ofstream fb(stem + "_bin2bucket.bin", std::ios::binary);
        fb.write(reinterpret_cast<const char*>(bin2bucket.data()),
                 sizeof(uint32_t) * bin2bucket.size());
    }
}

struct ThreadStats {
    int tid = -1;
    int last_bin = -1;  // 用于把连续 bin 压缩成区间
    uint64_t total_ns = 0;          // 本线程总耗时（累计各 bin）
    uint64_t total_bins = 0;        // 处理的 bin 数
    uint64_t total_bucket_ops = 0;  // sum(end - begin)

    // 将每个线程拿到的 bin 连续段做运行长度编码： [start, end] 闭区间
    std::vector<std::pair<uint32_t,uint32_t>> bin_ranges;

#if TRACE_PER_BIN
    std::vector<uint32_t> bins;                             
    std::vector<uint32_t> bin_bucket_counts;
    std::vector<uint64_t> bin_ns;
#endif
};

static void dump_thread_stats(const std::string& stem,  // 文件前缀（不含扩展名）
                              uint32_t B, int T,
                              int layer, int epoch, int iter,
                              const std::vector<ThreadStats>& stats) {

    // 线程汇总
    {
        std::ofstream out(stem + "_thread_summary.csv");
        out << "thread,bins,items,ns,ms,avg_ns_per_bin,avg_ns_per_item,ranges\n";
        for (int t = 0; t < T; ++t) {
            const auto& s = stats[t];
            const uint64_t bins = s.total_bins;
            const uint64_t items = s.total_bucket_ops;
            const uint64_t ns = s.total_ns;
            const double avg_bin = bins ? double(ns) / double(bins) : 0.0;
            const double avg_item = items ? double(ns) / double(items) : 0.0;
            out << t << ',' << bins << ',' << items << ','
                << ns << ',' << (ns/1e6) << ','
                << avg_bin << ',' << avg_item << ','
                << s.bin_ranges.size() << '\n';
        }
    }

    // 每线程的 bin 区间
    {
        std::ofstream out(stem + "_thread_ranges.csv");
        out << "thread,range_idx,bin_start,bin_end,bins_in_range\n";
        for (int t = 0; t < T; ++t) {
            const auto& s = stats[t];
            for (size_t k = 0; k < s.bin_ranges.size(); ++k) {
                auto [st,en] = s.bin_ranges[k];
                uint32_t len = (en >= st) ? (en - st + 1) : 0;
                out << t << ',' << k << ',' << st << ',' << en << ',' << len << '\n';
            }
        }
    }

#if TRACE_PER_BIN
    // 逐 bin 追踪（可选，体量大）
    {
        std::ofstream out(stem + "_thread_bins.csv");
        out << "thread,seq,bin,items,ns\n";
        for (int t = 0; t < T; ++t) {
            const auto& s = stats[t];
            for (size_t k = 0; k < s.bins.size(); ++k) {
                out << t << ',' << k << ',' << s.bins[k] << ','
                    << s.bin_bucket_counts[k] << ',' << s.bin_ns[k] << '\n';
            }
        }
    }
#endif

    // 元信息（简易 JSON，无第三方依赖）
    {
        std::ofstream out(stem + "_meta.json");
        out << "{\n"
            << "  \"B\": " << B << ",\n"
            << "  \"T\": " << T << ",\n"
            << "  \"layer\": " << layer << ",\n"
            << "  \"epoch\": " << epoch << ",\n"
            << "  \"iter\": " << iter << ",\n"
            << "  \"omp_max_threads\": " << omp_get_max_threads() << ",\n"
            << "  \"omp_dynamic\": " << omp_get_dynamic() << ",\n"
            << "  \"omp_proc_bind\": " << omp_get_proc_bind() << ",\n"
            << "  \"schedule\": \"static\"\n"
            << "}\n";
    }
}

Network::Network(int *sizesOfLayers, NodeType *layersTypes, int noOfLayers, int batchSize, float lr, int inputdim,  int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr) {

    _numberOfLayers = noOfLayers;
    _hiddenlayers = new Layer *[noOfLayers];
    _sizesOfLayers = sizesOfLayers;
    _layersTypes = layersTypes;
    _learningRate = lr;
    _currentBatchSize = batchSize;
    _Sparsity = Sparsity;

    for (int i = 0; i < noOfLayers; i++) {
        if (i != 0) {
            cnpy::NpyArray weightArr, biasArr, adamArr, adamvArr;
            float* weight, *bias, *adamAvgMom, *adamAvgVel;
            if(LOADWEIGHT){
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<float>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<float>();

                adamArr = arr["am_layer_"+to_string(i)];
                adamAvgMom = adamArr.data<float>();
                adamvArr = arr["av_layer_"+to_string(i)];
                adamAvgVel = adamvArr.data<float>();
            }

            std::cout<<"Start Init Layer 1"<<std::endl;
            std::time_t now = std::time(nullptr);
            std::cout << std::ctime(&now) << std::endl;

            _hiddenlayers[i] = new Layer(sizesOfLayers[i], sizesOfLayers[i - 1], i, _layersTypes[i], _currentBatchSize,  K[i], L[i], RangePow[i], Sparsity[i], weight, bias, adamAvgMom, adamAvgVel);
        } else {

            cnpy::NpyArray weightArr, biasArr, adamArr, adamvArr;
            float* weight, *bias, *adamAvgMom, *adamAvgVel;
            if(LOADWEIGHT){
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<float>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<float>();

                adamArr = arr["am_layer_"+to_string(i)];
                adamAvgMom = adamArr.data<float>();
                adamvArr = arr["av_layer_"+to_string(i)];
                adamAvgVel = adamvArr.data<float>();
            }

            std::cout<<"Start Init Layer 0"<<std::endl;
            std::time_t now = std::time(nullptr);
            std::cout << std::ctime(&now) << std::endl;

            _hiddenlayers[i] = new Layer(sizesOfLayers[i], inputdim, i, _layersTypes[i], _currentBatchSize, K[i], L[i], RangePow[i], Sparsity[i], weight, bias, adamAvgMom, adamAvgVel);
        }
    }
    cout << "after layer" << endl;
    std::time_t now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;
    _inputIDs = new int[_currentBatchSize]();
}

Layer *Network::getLayer(int LayerID) {
    if (LayerID < _numberOfLayers)
        return _hiddenlayers[LayerID];
    else {
        cout << "LayerID out of bounds" << endl;
    }
}

int Network::ProcessInput(int **inputIndices, float **inputValues, int *lengths, int **labels, int *labelsize, int iter, bool rehash, bool rebuild, int numBatch, int epoch, int* L) {

    float logloss = 0.0;
    int lowconf = 0;
    float prob = 0.0;
    int* avg_retrieval = new int[_numberOfLayers]();

    int check = 0;
    float tmplr = _learningRate;
    if (ADAM) {
        tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) /
                (1 - pow(BETA1, iter + 1));
    }else{
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    bool tmpRehash;
    bool tmpRebuild;

    ofstream LSHfile;
    if(isLog == 1){
        if(isHopper == 1){
            LSHfile.open("/scratch/zqu4/SLIDE-MP/dataset/LSH-epoch-"+ to_string(epoch)+".txt",  std::ios_base::app);
        }else{
            LSHfile.open("../dataset/LSH-epoch-"+ to_string(epoch)+".txt",  std::ios_base::app);
        }
    }

    std::cout << "Start Update" <<std::endl;
    std::time_t now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

    for (int l=1; l<_numberOfLayers ;l++) {
        if(rehash & _Sparsity[l]<1){
            tmpRehash=true;
        }else{
            tmpRehash=false;
        }
        if(rebuild & _Sparsity[l]<1){
            tmpRebuild=true;
        }else{
            tmpRebuild=false;
        }
        if (tmpRehash) {
            if(isLog == 1){
                LSHfile << "Layer ID : " << _hiddenlayers[l]->_layerID << endl;
            }

        }
        if (tmpRebuild){
            _hiddenlayers[l]->updateTable();
        }
        int ratio = 1;

        std::cout << "Start Neurons update" <<std::endl;
        std::time_t now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        if (l != 0) {
            if (tmpRehash) {
                std::cout << "Start LSH Rehash update" << std::endl;
                std::time_t now = std::time(nullptr);
                std::cout << std::ctime(&now) << std::endl;
            }

            if (tmpRehash) {

                std::cout << "Before erase, _AllNodes.size(): " << _hiddenlayers[l]->_AllNodes.size() << std::endl;
                size_t numofErase = (1<<_hiddenlayers[l]->_RangePow)*BUCKETSIZE;
                std::cout << "Nodes to erase: " << numofErase << std::endl;

                auto first_to_erase = _hiddenlayers[l]->_AllNodes.begin();
                auto last_to_erase = _hiddenlayers[l]->_AllNodes.begin() + numofErase;

                _hiddenlayers[l]->_AllNodes.erase(first_to_erase, last_to_erase);

                std::cout << "After erase, _AllNodes.size(): " << _hiddenlayers[l]->_AllNodes.size() << std::endl;

                std::cout << "Step 1: Calculating total nodes to add from LSH..." << std::endl;
                size_t totalNodesToAdd = 0;
                for (int i = 0; i < (1 << _hiddenlayers[l]->_RangePow); i++) {
                    Bucket *tmpbuc = &(_hiddenlayers[l]->_hashTables->_bucket[0][i]);
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        if (tmpbuc->arr[j] != nullptr) {
                            totalNodesToAdd++;
                        }
                    }
                }
                std::cout << "Total nodes found in LSH: " << totalNodesToAdd << std::endl;
                if (totalNodesToAdd != numofErase){
                    std::cout << "size mismatch!!!!!!!! ERROR!!!!!!" <<std::endl;
                    exit(1);
                }

                // 步骤 2: 使用 reserve 为 _AllNodes 预留足够的内存空间
                std::cout << "Step 2: Reserving space in _AllNodes..." << std::endl;
                std::cout << "Current _AllNodes size: " << _hiddenlayers[l]->_AllNodes.size() << std::endl;
                std::cout << "New capacity will be at least: " << _hiddenlayers[l]->_AllNodes.size() + totalNodesToAdd << std::endl;

                _hiddenlayers[l]->_AllNodes.reserve(_hiddenlayers[l]->_AllNodes.size() + totalNodesToAdd);

                std::cout << "New _AllNodes capacity: " << _hiddenlayers[l]->_AllNodes.capacity() << std::endl;

                std::cout << "Step 3: Adding nodes from LSH back to _AllNodes..." << std::endl;

                for (int i = 0; i < (1 << _hiddenlayers[l]->_RangePow); i++) {
                    Bucket *tmpbuc = &(_hiddenlayers[l]->_hashTables->_bucket[0][i]);

                    for (int j = 0; j < BUCKETSIZE; j++) {
                        if (tmpbuc->arr[j] != nullptr) {
                            _hiddenlayers[l]->_AllNodes.push_back(*(tmpbuc->arr[j]));
                        }
                    }
                }

                std::cout << "Step 3 complete. Final _AllNodes size: " << _hiddenlayers[l]->_AllNodes.size() << std::endl;

                _hiddenlayers[l]->_hashTables->clear();

                for (int row = 0; row < _hiddenlayers[l]->_L; ++row) {
                    const uint32_t M = 1u << _hiddenlayers[l]->_RangePow;
                    for (uint32_t j = 0; j < M; ++j) {
                        _hiddenlayers[l]->_hashTables->_bucket[row][j]._bucID = (int)j;
                    }
                }

                std::cout << "LSH Rehash update: 1st linear scan to update lsh signatures" << std::endl;
                std::time_t now = std::time(nullptr);
                std::cout << std::ctime(&now) << std::endl;

                for (Node &node: _hiddenlayers[l]->_AllNodes) {
                    if (tmpRehash) {
                        int *hashes;

                        hashes = _hiddenlayers[l]->_dwtaHasher->getHashEasy(node._weights, node._dim, TOPK);
                        int *hashIndices = _hiddenlayers[l]->_hashTables->hashesToIndex(hashes);

                        int pred = obl::OblLt(node.isDummy, 1);

                        node._lshIndex = obl::OblChoose(pred, hashIndices[0], node._lshIndex);
                    }
                }

                // 1) First comparator: sort by _lshIndex ascending, then by _isDummy
                auto cmpFirst = [](const Node &a, const Node &b) {

                    // 1) 比较 lshIndex：谁小谁应该在前
                    int idx_lt = obl::OblLt(a._lshIndex, b._lshIndex);
                    int idx_gt = obl::OblLt(b._lshIndex, a._lshIndex);
                    int idx_eq = 1 - idx_lt - idx_gt;

                    // 2) 比较 isDummy：real(0) 应该在 dummy(>0) 前面，相当于升序
                    int dummy_lt = obl::OblLt(a.isDummy, b.isDummy);

                    // 3) 如果 index 不相等，用 idx_lt；
                    //    如果 index 相等，用 dummy_lt。
                    int res_bit = obl::OblChoose(idx_eq, dummy_lt, idx_lt);

                    return res_bit != 0;
                };

                std::cout << "LSH Rehash update: 1st o_sort" << std::endl;
                now = std::time(nullptr);
                std::cout << std::ctime(&now) << std::endl;

                _hiddenlayers[l]->xorBitonicSort_Parallel(_hiddenlayers[l]->_AllNodes, cmpFirst);

                //            identify the overflow ones withing the same LSH index
                int currentIndex = -1;
                int countForIndex = 0;

                std::cout << "LSH Rehash update: 2nd linear scan to identify exceed nodes" << std::endl;
                now = std::time(nullptr);
                std::cout << std::ctime(&now) << std::endl;

                for (auto &node: _hiddenlayers[l]->_AllNodes) {

                     // 是否“同一个 index”：1 表示相等，0 表示不等
                    int sameIndex = obl::OblEq(node._lshIndex, currentIndex);
                    int isNew     = 1 - sameIndex;

                    int updatedCurrent = obl::OblChoose(isNew, node._lshIndex, currentIndex);
                    // branchless 更新 count（在判断 overflow 之前应该是 0 或旧的 countForIndex）
                    int baseCount      = obl::OblChoose(isNew, 0, countForIndex);

                    // 根据 baseCount 判定 overflow：前 BUCKETSIZE 个是 non-overflow
                    int nonOverflow    = obl::OblLt(baseCount, BUCKETSIZE);
                    int overflowFlag   = 1 - nonOverflow;

                    node._isOverflow   = overflowFlag;

                    currentIndex  = updatedCurrent;
                    countForIndex = baseCount + 1;
                }

                auto finalComparator = [](const Node &a, const Node &b) {

                    // 1) overflow 标志：non-overflow(0) < overflow(1)
                    int of_lt = obl::OblLt(a._isOverflow, b._isOverflow);
                    int of_eq = obl::OblEq(a._isOverflow, b._isOverflow);

                    // 2) 在相同 overflow 里，按 lshIndex 升序
                    int idx_lt = obl::OblLt(a._lshIndex, b._lshIndex);
                    int idx_eq = obl::OblEq(a._lshIndex, b._lshIndex);

                    // 3) 在相同 overflow + 相同 lshIndex 里，real(0) < dummy(1)
                    int dum_lt = obl::OblLt(a.isDummy, b.isDummy);

                    // 第一层：如果 overflow 不同，用 of_lt 决定；如果相同，先用 idx_lt。
                    int res1 = of_lt;
                    int res2 = obl::OblChoose(of_eq, idx_lt, res1);

                    // 只有在 overflow 和 lshIndex 都相等时，才用 dummy 来打破平局
                    int both_eq = of_eq & idx_eq;
                    int res3    = obl::OblChoose(both_eq, dum_lt, res2);

                    return res3 != 0;
                };

                std::cout << "LSH Rehash update: 2nd sort to move exceed nodes" << std::endl;
                now = std::time(nullptr);
                std::cout << std::ctime(&now) << std::endl;

                _hiddenlayers[l]->xorBitonicSort_Parallel(_hiddenlayers[l]->_AllNodes, finalComparator);

                std::cout << "LSH Rehash update: 3rd linear scan to write into LSH table" << std::endl;
                now = std::time(nullptr);
                std::cout << std::ctime(&now) << std::endl;

                size_t nodeIndex = 0;

                for (int i = 0; i < (1 << _hiddenlayers[l]->_RangePow); i++) {
                    Bucket *tmpbuc = &(_hiddenlayers[l]->_hashTables->_bucket[0][i]);
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        if (nodeIndex < _hiddenlayers[l]->_AllNodes.size()) {
                            if (tmpbuc->arr[j] == nullptr) {
                                tmpbuc->arr[j] = new Node(_hiddenlayers[l]->_AllNodes[nodeIndex]);
                                nodeIndex++;
                            } else {
                                *(tmpbuc->arr[j]) = _hiddenlayers[l]->_AllNodes[nodeIndex];
                                nodeIndex++;
                            }

                        } else {
                            std::cout << "LSH run out of ALL Node List, ERROR" << std::endl;
                        }

                    }
                }

            }

        }

        _hiddenlayers[l]->isLSHinit = true;

    }

    if(isLog == 1){
        LSHfile.close();
    }

// process Input Layer with the first hidden layer (hidden_laer[0])

    std::cout << "Start FF" <<std::endl;
    now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

    int ***activenodesperlayer = new int **[_currentBatchSize]();
    float ***activeValuesperlayer = new float **[_currentBatchSize]();
    int **sizes = new int *[_currentBatchSize]();

    for (int i = 0; i < _currentBatchSize; i++) {
        activenodesperlayer[i] = new int * [_numberOfLayers]();
        activeValuesperlayer[i] = new float * [_numberOfLayers]();
        sizes[i] = new int [_numberOfLayers]();
    }

    std::cout << "Start FF: layer 0 " <<std::endl;
    now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;
    for(int i = 0; i < _currentBatchSize; i++) {
        int in;

        auto t1 = std::chrono::high_resolution_clock::now();
        in = _hiddenlayers[0]->ComputeActivation_FirstLayer(inputIndices, inputValues, lengths, activenodesperlayer, activeValuesperlayer, sizes, 0, i, _Sparsity[0]);
        avg_retrieval[0] += in;

    }

    // implicitly require at least 2 layers
    for (int j = 1; j < _numberOfLayers; j++) {
        std::cout << "Start FF: layer  " << j <<std::endl;
        std::time_t now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;
        // batch query, to get a batch of inputs' Multi-Probe hash signatures in parallel
        // for generating batch requests
        int ** requestIDs = new int * [_currentBatchSize]();

        int K = _hiddenlayers[j]->_K; // K is the number of LSH hash functions at layer j

        // calculate number of LSH bucket that need to be retrieved based on the MP-WTA
        // basically, K LSH functions will give us K signatures, MP-WTA will choose some of signatures and replace them with Max-2 or Max-3 values
        // so, choose 1 of K and replace with Max-2 or Max-3 = K + K
        // choose 2 of K and replace with Max-2 or Max-3 = [ K * (K-1) / 2 ] * (2 * 2)
        // choose 3 of K and replace with Max-2 or Max-3 = { [ K * (K - 1) * (K - 2) ] / [3*2] } * (2*2*2)
        int numReq = K * (K - 1) * (K - 2) / (3*2) * (2*2*2) + K * (K - 1) / (2) * (2*2) + K + K + 1;
        int totalReq = numReq * _currentBatchSize;

        for (int i = 0; i < _currentBatchSize; i ++) {
            requestIDs[i] = new int[numReq]();
        }

        std::cout << "Start FF: layer  " << j <<  " , compute Request Buc IDs"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        # pragma omp parallel for
        for (int i = 0; i < _currentBatchSize; i++){
            _hiddenlayers[j]->query(activenodesperlayer, activeValuesperlayer, sizes, j-1, i, _Sparsity[j], requestIDs);

        }

        std::cout << "Start FF: layer  " << j <<  " , generate Requests"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;
        if (_hiddenlayers[j]->_requester == nullptr
            ||_hiddenlayers[j]->_requester->_reqTable == nullptr){
            std::cout << "Re-init Requester" << std::endl;
            now = std::time(nullptr);
            std::cout << std::ctime(&now) << std::endl;
            _hiddenlayers[j]->_requester = new Requester(totalReq);
        }
        for (int i = 0; i < _currentBatchSize; i++) {
            // 注意，这里generate出来的requestBuc里面是不包含dummy node的， 全是Nullptr
            _hiddenlayers[j]->_requester->generateRequestBucs(requestIDs[i], numReq, i, _hiddenlayers[j-1]->_noOfNodes, _hiddenlayers[j-1]->_type, j, _currentBatchSize);

        }

        std::cout << "Start FF: layer  " << j <<  " , generate Requests : "  << _hiddenlayers[j]->_requester->_currentReqIndex <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        std::cout << "Start FF: layer  " << j <<  " , preparning oHT"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;
        // all requests buckets of a batch inputs are generated
        // now, transfer the request table to a Two-Tier Hash Table (OHT)
        std::cout <<"Total requests : " << totalReq << std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;
        for (int i = 0; i < totalReq; i++) {
            RequestBuc  e ;
            if (_hiddenlayers[j]->_requester->_reqTable[i] == nullptr){
                std::cout << "Request is nullprt" <<std::endl;
            }
            e = *_hiddenlayers[j]->_requester->_reqTable[i];

            _hiddenlayers[j]->_oht->prepare(std::move(e));
            if(_hiddenlayers[j]->_requester->_reqTable[i] == nullptr || _hiddenlayers[j]->_requester->_reqTable[i] == NULL){
                std::cout<<"The Request in ReqTable is null after generating OHT, ERROR 1" <<std::endl;
            }
        }

        std::cout << "Start FF: layer  " << j <<  " , build OHT"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        _hiddenlayers[j]->_oht->build();

        // here, the OHT is built without any padded dummy neurons, need to pad
        std::cout << "Start FF: layer  " << j <<  " , padding OHT"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;
        _hiddenlayers[j]->_oht->paddingnodes( _hiddenlayers[j-1]->_noOfNodes, _hiddenlayers[j-1]->_type, j, _currentBatchSize);

        std::cout << "Start FF: layer  " << j <<  " , OHT access by LSH"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;
        // now, linear scan each bucket of LSH, and lookup it in OHT
        // remember to do oblivious copy

        // 假设：已完成 OHT build()，bins1_/bins2_ 定长，不再 push_back（避免并发 reallocate）

        if(PARALLEL){
            auto t_1 = std::chrono::high_resolution_clock::now();
            const uint32_t B = _hiddenlayers[j]->_oht->kB_;
            const uint32_t N = 1u << _hiddenlayers[j]->_RangePow;

            std::vector<std::vector<uint32_t>> bin_lists(B);
            for (uint32_t i = 0; i < N; i++) {

                const int bucID = _hiddenlayers[j]->_hashTables->_bucket[0][i]._bucID;
                const uint32_t b = std::hash<int>{}(bucID) % B;

                if(i != bucID){
                    std::cout << "Bucket ID does not equal to i, ERROR!" <<std::endl;
                    std::cout << "Bucket iD is " << bucID << ", i is :" << i << std::endl;
                    exit(0);
                }

                bin_lists[b].push_back(i);
            }

// 计算 starts 并拼接为扁平数组
           std::vector<uint32_t> starts(B + 1, 0);
           for (uint32_t b = 0; b < B; ++b) starts[b + 1] = starts[b] + (uint32_t)bin_lists[b].size();

           std::vector<uint32_t> bin2bucket(starts.back());
           for (uint32_t b = 0; b < B; ++b) {
                if (!bin_lists[b].empty())
                    std::memcpy(&bin2bucket[starts[b]],
                                bin_lists[b].data(),
                                bin_lists[b].size() * sizeof(uint32_t));
           }

            // 生成一个带时间戳的前缀，避免覆盖
            auto now = std::chrono::system_clock::now();
            std::time_t tt = std::chrono::system_clock::to_time_t(now);
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &tt);
#else
            localtime_r(&tt, &tm);
#endif
            std::string stem = "../dataset/binmaps/" + std::to_string(B) + "-" + std::to_string(THREATNUM)
                    + "-" + std::to_string(epoch) + "-" + std::to_string(iter);
            dump_bin_layout(stem, B, starts, bin2bucket);

            omp_set_dynamic(0);

            // 读取外部配置（0 => 用最大可用）
            int req = THREATNUM;
            if (req <= 0) req = omp_get_max_threads();

            const int T = std::max(1, std::min(req, omp_get_max_threads()));

            int printed = 0;

            auto t_2 = std::chrono::high_resolution_clock::now();

            const int T_used = T;
            std::vector<ThreadStats> thr_stats(T_used);

            using Clock = std::chrono::steady_clock;

            int willEmpty = 0;
            if (B > N){
                willEmpty = N;
            } else{
                willEmpty = B;
            }

#pragma omp parallel for schedule(static) num_threads(T)
            for (int b = 0; b < (int)willEmpty; b++) {
                if (!printed) {
                    #pragma omp critical
                    if (!printed) {
                        printed = 1;
                        fprintf(stderr,
                                "[pass4] threads=%d  (max=%d, dynamic=%d, proc_bind=%d)\n",
                                omp_get_num_threads(),
                                omp_get_max_threads(),
                                omp_get_dynamic(),
                                omp_get_proc_bind());
                    }
                }

                int tid = omp_get_thread_num();
                auto& ts = thr_stats[tid];
                if (ts.tid < 0) ts.tid = tid;

                // 维护连续区间
                if (ts.bin_ranges.empty() || ts.last_bin + 1 != b) {
                    ts.bin_ranges.emplace_back((uint32_t)b, (uint32_t)b);
                } else {
                    ts.bin_ranges.back().second = (uint32_t)b;
                }
                ts.last_bin = b;

                const uint32_t begin = starts[b];
                const uint32_t end   = starts[b + 1];

                auto t0 = Clock::now();

                for (uint32_t off = begin; off < end; off++) {
                    const uint32_t idx = bin2bucket[off];
                    auto &lsh_bucket = _hiddenlayers[j]->_hashTables->_bucket[0][idx];
                    _hiddenlayers[j]->_oht->lookupLSH_write(lsh_bucket);   // 每次都会扫描该 bin 的 kZ_ 槽位
                }

                auto t1 = Clock::now();
                uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

                ts.total_ns        += ns;
                ts.total_bins      += 1;
                ts.total_bucket_ops += (end - begin);

#if TRACE_PER_BIN
                ts.bins.push_back((uint32_t)b);
                ts.bin_bucket_counts.push_back(end - begin);
                ts.bin_ns.push_back(ns);
#endif

            }

            auto t_3 = std::chrono::high_resolution_clock::now();

            std::vector<std::vector<uint32_t>>().swap(bin_lists);
            std::vector<uint32_t>().swap(starts);
            std::vector<uint32_t>().swap(bin2bucket);

            int tf_1 = std::chrono::duration_cast<std::chrono::milliseconds>(t_2 - t_1).count();
            int tf_2 = std::chrono::duration_cast<std::chrono::milliseconds>(t_3 - t_2).count();
            int tf_3 = std::chrono::duration_cast<std::chrono::milliseconds>(t_3 - t_1).count();

            std::cout << "time of building scheduler : " << tf_1/1000 << std::endl;
            std::cout << "time of parallel: " << tf_2/1000 << std::endl;
            std::cout << "time of total : " << tf_3/1000 << std::endl;

            std::string stemmp = "../dataset/binmaps/" + std::to_string(B) + "-" + std::to_string(THREATNUM)
                               + "-" + std::to_string(epoch) + "-" + std::to_string(iter);
            dump_thread_stats(stemmp, B, T_used, j, epoch, iter, thr_stats);

        }
        else{
            // here we assume there always only 1 LSH table
            std::cout << "size of hash table : " << (1 << _hiddenlayers[j]->_RangePow) << std::endl;
            for (int i = 0; i < (1 << _hiddenlayers[j]->_RangePow); i++) {
    //              In Elains'paper, it requires the keys of input set for build and query be distinct, even if it use standard padding and oblivious sort
    //              The possible reason is -- elain read content from OHT to query, so if there are identical keys, it cannot decide which one should be found for the query
    //              In our case, we are using query to write into OHT, so the multiple elements with the same keys in the same bin won't be an issue
    //              cause we will linear scan the entire bin and each elements and do Obliivous copy on each of the elements
    //              Remember to CORRECTLY implement the lookup code here, because in the next lookup for OHT->Requester,
    //                  we are using OHT to write into Requester, the order is flipped
                _hiddenlayers[j]->_oht->lookupLSH_write(_hiddenlayers[j]->_hashTables->_bucket[0][i]);
            }
        }

        std::cout << "Start FF: layer  " << j <<  " , OHT access by Requester"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        // OHT bins1_/bins2_ now hold real entries (filled by lookupLSH_read) plus
        // dummies/fillers, each tier sorted by (Excess, bin_id, dummy/filler) from
        // build(). Merge both tiers and re-sort once by (real-first, _inputRank) so
        // the top `totalReq` slots of the merged array (= the front of bins1_ after
        // the split) are the real entries grouped by input rank, then
        // oblivious-deep-copy them back into _reqTable in order.
          
        // Sorting the two tiers as one unit follows the two-tier OHT paper. Under
        // the configured (kB_, kZ_) no real entry overflows into bins2_, so this is
        // equivalent to sorting bins1_ alone; it stays correct if that ever changes,
        // because real entries from either tier are gathered to the front.
        const auto cmpReadback = [](const RequestBuc &a, const RequestBuc &b) {
            // real (tag==0) -> key = _inputRank; non-real -> key = INT_MAX (pushed to back)
            int aKey = obl::OblChoose(obl::OblEq(a.tag, 0), a._inputRank, 0x7FFFFFFF);
            int bKey = obl::OblChoose(obl::OblEq(b.tag, 0), b._inputRank, 0x7FFFFFFF);
            return obl::OblLt(aKey, bKey);
        };
        _hiddenlayers[j]->_oht->sort_tiers_merged(cmpReadback);
        for (int k = 0; k < totalReq; k++) {
            RequestBuc::obl_copy_reqbuc(
                    true,
                    &_hiddenlayers[j]->_oht->bins1_[k],
                    _hiddenlayers[j]->_requester->_reqTable[k]);
        }

        // now, all neurons are read from LSH, stored in Requests Table
        for (int i = 0; i < _currentBatchSize; i++){
            activenodesperlayer[i][j] = new int  [BUCKETSIZE * numReq]();
            activeValuesperlayer[i][j] = new float  [BUCKETSIZE * numReq]();
            sizes[i][j] = BUCKETSIZE * numReq;
        }

        std::cout << "Start FF: layer  " << j <<  " , compute activation"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;
        # pragma  omp parallel for
        for (int i = 0; i < _currentBatchSize; i++) {
            float maxValue = 0;
            if (_hiddenlayers[j]->_type == NodeType::Softmax)
                _hiddenlayers[j]->_normalizationConstants[i] = 0;

            for (int k = i * numReq; k < (i + 1) * numReq; k++) {
                for (int n = 0; n < BUCKETSIZE; n ++) {
                    Node * tmpnode = _hiddenlayers[j]->_requester->_reqTable[k]->arr[n];
                    // if a neuron is dummy, then its ID = -1, its activation = 0
                    float activation = tmpnode->getActivation(activenodesperlayer[i][j-1], activeValuesperlayer[i][j-1], sizes[i][j-1], i);
                    activenodesperlayer[i][j][n] = tmpnode->_IDinLayer;
                    activeValuesperlayer[i][j][n] = activation;

                    if(_hiddenlayers[j]->_type == NodeType::Softmax){
                       int flag = obl::OblGt(activeValuesperlayer[i][j][n], maxValue);
                       maxValue = obl::OblChoose(flag, activeValuesperlayer[i][j][n], maxValue);

                    }
                }
            }

            if(_hiddenlayers[j]->_type == NodeType::Softmax) {
                for (int k = i * numReq; k < (i + 1) * numReq; k++) {
                    for (int n = 0; n < BUCKETSIZE; n++) {
                        Node * tmpnode = _hiddenlayers[j]->_requester->_reqTable[k]->arr[n];

                        float realActivation = exp(activeValuesperlayer[i][j][n] - maxValue);
                        activeValuesperlayer[i][j][n] = realActivation;
                        tmpnode->_lastActivations[i] = realActivation;
                        _hiddenlayers[j]->_normalizationConstants[i] += realActivation;

                    }
                }
            }

        }

        for (int i = 0; i < _currentBatchSize; i++) {
            delete[] requestIDs[i];
        }

        delete[] requestIDs;

        requestIDs = nullptr;

    }

//    Now, do back propagation with snoopy
    std::cout << "Start BP" << std::endl;
    now = std::time(nullptr);
    std::cout << std::ctime(&now) << std::endl;

    for (int j = _numberOfLayers - 1; j >= 0; j--) {
        std::cout << "Start BP: layer  " << j <<std::endl;
        std::time_t now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        int K = _hiddenlayers[j]->_K;
        int numReq = K * (K - 1) * (K - 2) / (3*2) * (2*2*2) + K * (K - 1) / (2) * (2*2) + K + K + 1;
        int totalReq = numReq * _currentBatchSize;
         for (int i = 0; i < _currentBatchSize; i++) {
            if (j == _numberOfLayers - 1) {
                for (int k = i * numReq; k < (i + 1) * numReq; k++) {
                    for (int n = 0; n < BUCKETSIZE; n++) {
                        Node * tmpnode = _hiddenlayers[j]->_requester->_reqTable[k]->arr[n];
                        tmpnode->ComputeExtaStatsForSoftMax(_hiddenlayers[j]->getNomalizationConstant(i),i, labels[i], labelsize[i]);
                    }
                }
            }

            if (j != 0) {
                int Kprev = _hiddenlayers[j-1]->_K;
                int numReqprev = Kprev * (Kprev - 1) * (Kprev - 2) / (3*2) * (2*2*2) + Kprev * (Kprev - 1) / (2) * (2*2) + Kprev + Kprev + 1;
                int totalReqprev = numReqprev * _currentBatchSize;

                for (int k = i * numReq; k < (i + 1) * numReq; k++) {
                    for (int n = 0; n < BUCKETSIZE; n++) {
                        Node * tmpnode = _hiddenlayers[j]->_requester->_reqTable[k]->arr[n];

                        if (j != 1){
                            for (int kprev = i * numReqprev; kprev < (i + 1) * numReqprev; kprev++) {
                                for (int m = 0; m < BUCKETSIZE; m++) {
                                    Node * tmpnodeprev = _hiddenlayers[j-1]->_requester->_reqTable[kprev]->arr[m];

                                    tmpnode->backPropageSingle(tmpnodeprev, _learningRate, i);
                                }
                            }
                        }
                        else {
                            for (int kprev = 0; kprev < _hiddenlayers[j-1]->_noOfNodes; kprev ++) {
                                Node * tmpnodeprev = _hiddenlayers[j-1]->getNodebyID(kprev);
                                tmpnode->backPropageSingle(tmpnodeprev, _learningRate, i);
                            }
                        }

                        tmpnode->backPropageBias(_learningRate, i);

                        tmpnode->clearForInput(i);

                    }
                }
            } else {
                // deal with first layer's backpropagation
                for (int n = 0; n < _hiddenlayers[j]->_noOfNodes; n++){
                    _hiddenlayers[j]->getNodebyID(n)->backPropagateFirstLayer(inputIndices[i], inputValues[i], lengths[i], tmplr, i);
                }
            }
        }
    }

    std::cout << "Free Memory"  <<std::endl;
    for (int i = 0; i < _currentBatchSize; i++) {
        for (int j = 0; j < _numberOfLayers; j++) {
            if (activenodesperlayer[i][j] != nullptr)
                delete[] activenodesperlayer[i][j];
            if (activeValuesperlayer[i][j] != nullptr)
                delete[] activeValuesperlayer[i][j];
        }
        if (sizes[i] != nullptr)
            delete[] sizes[i];
        delete[] activenodesperlayer[i];
        delete[] activeValuesperlayer[i];
    }
    delete[] activenodesperlayer;
    delete[] activeValuesperlayer;

    int ratio = 1;
    for (int l = 1; l < _numberOfLayers; l++) {
        std::cout << "Start Requester sort by ID"  <<std::endl;
        std::time_t now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;
        _hiddenlayers[l]->_requester->sortByIDBitonicPtr();

        int totalReqMerge = _hiddenlayers[l]->_requester->_totalReqs;

        std::cout << "Start Requester Merge"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;
        _hiddenlayers[l]->_requester->obl_mergeSameBucID(totalReqMerge);

        std::cout << "End Requester Merge"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        std::cout << "Start Neuron ADAM update"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        for (int i = 0; i < totalReqMerge; i++) {
            for (int j = 0; j < BUCKETSIZE; j++) {
                Node * tmp = _hiddenlayers[l]->_requester->_reqTable[i]->arr[j];

                if(ADAM){
                    for (int d=0; d< tmp->_dim;d++){
                        tmp->_adamAvgMom[d] = BETA1 * tmp->_adamAvgMom[d] + (1 - BETA1) * tmp->_t[d];
                        tmp->_adamAvgVel[d] = BETA2 * tmp->_adamAvgVel[d] + (1 - BETA2) * tmp->_t[d] * tmp->_t[d];
                        tmp->_weights[d] += ratio * tmplr * tmp->_adamAvgMom[d] / (sqrt(tmp->_adamAvgVel[d]) + EPS);
                        tmp->_t[d] = 0;
                    }

                    tmp->_adamAvgMombias = BETA1 * tmp->_adamAvgMombias + (1 - BETA1) * tmp->_tbias;
                    tmp->_adamAvgVelbias = BETA2 * tmp->_adamAvgVelbias + (1 - BETA2) * tmp->_tbias * tmp->_tbias;
                    tmp->_bias += ratio * tmplr * tmp->_adamAvgMombias / (sqrt(tmp->_adamAvgVelbias) + EPS);
                    tmp->_tbias = 0;

                }
                else{
                    std::copy(tmp->_mirrorWeights, tmp->_mirrorWeights+(tmp->_dim) , tmp->_weights);
                    tmp->_bias = tmp->_mirrorbias;
                }
            }
        }

        std::cout << "End Neuron ADAM update"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        std::cout << "Start write Requester back to LSH"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        // O3 OHT Layout Reuse
        // Skip the full OHT rebuild (clear + per-request prepare + build with two
        // bitonic sorts). After §8 read-back, _oht->bins1_[0..totalReqMerge-1]
        // holds the FF-time real entries (sorted by inputRank) and bins1_[totalReqMerge..end]
        // holds the FF-time non-real entries; every entry carries its FF-time
        // oht_pos. §9 merge mutated _reqTable but did not touch oht_pos.
          
        // Fast rebuild: overwrite bins1_[0..totalReqMerge-1] with the merged
        // _reqTable contents (oht_pos values are unchanged because they were copied
        // from bins1_ in §8 and merge does not write oht_pos), then merge both tiers
        // and run a single oblivious sort by oht_pos. build_pass assigned bins1_
        // oht_pos in [0, kB_*kZ_) and bins2_ in [kB_*kZ_, 2*kB_*kZ_), so the sorted
        // array splits back to the exact FF layout of both tiers with no ties.
        // The O2 scheduler below is rebuilt independently and does not depend on
        // bins1_/bins2_ contents.

        std::cout <<"Total requests : " << totalReqMerge << std::endl;
        std::cout << "Start O3 fast OHT rebuild"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        {
            auto* oht_l = _hiddenlayers[l]->_oht;
            auto** reqTable_l = _hiddenlayers[l]->_requester->_reqTable;
            for (int k = 0; k < totalReqMerge; k++) {
                RequestBuc::obl_copy_reqbuc(true, reqTable_l[k], &oht_l->bins1_[k]);
            }
            const auto cmp_oht_pos = [](const RequestBuc &a, const RequestBuc &b) {
                return obl::OblLt(a.oht_pos, b.oht_pos);
            };
            oht_l->sort_tiers_merged(cmp_oht_pos);
        }

        std::cout << "End O3 fast OHT rebuild"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        std::cout << "Start access by LSH"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        if (PARALLEL){
            using Clock = std::chrono::steady_clock;

            std::time_t t = std::time(nullptr);
            std::cout << std::asctime(std::localtime(&t)) << std::endl;

            auto t_1 = Clock::now();

            const uint32_t B = _hiddenlayers[l]->_oht->kB_;
            const uint32_t N = 1u << _hiddenlayers[l]->_RangePow;

// 构建调度表（单遍）
            std::vector<std::vector<uint32_t>> bin_lists(B);

            for (uint32_t i = 0; i < N; ++i) {
                const int bucID = _hiddenlayers[l]->_hashTables->_bucket[0][i]._bucID;
                const uint32_t b = std::hash<int>{}(bucID) % B;

                if(i != bucID){
                    std::cout << "Bucket ID does not equal to i, ERROR!" <<std::endl;
                    std::cout << "Bucket iD is " << bucID << ", i is :" << i << std::endl;
                    exit(0);
                }
                bin_lists[b].push_back(i);
            }

            std::vector<uint32_t> starts(B + 1, 0);
            for (uint32_t b = 0; b < B; ++b) starts[b + 1] = starts[b] + (uint32_t)bin_lists[b].size();
            
            std::vector<uint32_t> bin2bucket(starts.back());
            for (uint32_t b = 0; b < B; ++b) {
                if (!bin_lists[b].empty())
                    std::memcpy(&bin2bucket[starts[b]], bin_lists[b].data(),
                                bin_lists[b].size() * sizeof(uint32_t));
            }

            auto t_2 = Clock::now();
            
            omp_set_dynamic(0);

            int req = THREATNUM;
            if (req <= 0) req = omp_get_max_threads();
            const int T = std::max(1, std::min(req, omp_get_max_threads()));

            // 对齐第一段：若 B > N，只处理到 min(B, N)
            const uint32_t willEmpty = (B > N) ? N : B;

            int printed = 0;

#pragma omp parallel for schedule(static) num_threads(T)
            for (int b = 0; b < (int)willEmpty; ++b) {
                if (!printed) {
        #pragma omp critical
                    if (!printed) {
                        printed = 1;
                        fprintf(stderr,
                                "[readback] threads=%d  (max=%d, dynamic=%d, level=%d, in_parallel=%d, proc_bind=%d)\n",
                                omp_get_num_threads(),
                                omp_get_max_threads(),
                                omp_get_dynamic(),
                                omp_get_level(),
                                omp_in_parallel(),
                                omp_get_proc_bind());
                    }
                }

                const uint32_t begin = starts[(uint32_t)b];
                const uint32_t end   = starts[(uint32_t)b + 1];

                for (uint32_t off = begin; off < end; ++off) {
                    const uint32_t idx = bin2bucket[off];
                    auto &lsh_bucket = _hiddenlayers[l]->_hashTables->_bucket[0][idx];
                    _hiddenlayers[l]->_oht->lookupLSH_read(lsh_bucket);
                }
            }

            auto t_3 = Clock::now();
                    
            std::vector<std::vector<uint32_t>>().swap(bin_lists);
            std::vector<uint32_t>().swap(starts);
            std::vector<uint32_t>().swap(bin2bucket);

            auto t_4 = Clock::now();

            _hiddenlayers[l]->_oht->clear();

            auto t_5 = Clock::now();

            auto ms_sched = std::chrono::duration_cast<std::chrono::milliseconds>(t_2 - t_1).count();
            auto ms_loop  = std::chrono::duration_cast<std::chrono::milliseconds>(t_3 - t_2).count();
            auto ms_free  = std::chrono::duration_cast<std::chrono::milliseconds>(t_4 - t_3).count();
            auto ms_clear = std::chrono::duration_cast<std::chrono::milliseconds>(t_5 - t_4).count();
            auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t_5 - t_1).count();

            std::cout << "LSH->Requester(readback) time breakdown (sec):" << std::endl;
            std::cout << "  scheduler : " << (ms_sched / 1000.0) << std::endl;
            std::cout << "  parallel  : " << (ms_loop  / 1000.0) << std::endl;
            std::cout << "  free      : " << (ms_free  / 1000.0) << std::endl;
            std::cout << "  oht.clear : " << (ms_clear / 1000.0) << std::endl;
            std::cout << "  total     : " << (ms_total / 1000.0) << std::endl;

        }
        else{
            for (int i = 0; i < (1 << _hiddenlayers[l]->_RangePow); i++) {
    //            This time, the query is stil LSH bucket, but LSH bucket read content from oht,
    //            which means, write oht content into LSH bucket.
                _hiddenlayers[l]->_oht->lookupLSH_read(_hiddenlayers[l]->_hashTables->_bucket[0][i]);
            }

            _hiddenlayers[l]->_oht->clear();

        }

        std::cout << "End write Requester back to LSH"  <<std::endl;
        now = std::time(nullptr);
        std::cout << std::ctime(&now) << std::endl;

        delete _hiddenlayers[l]->_requester;
        _hiddenlayers[l]->_requester = nullptr;

        std::cout << "End delete Requester"  << std::endl;

    }

    auto t2 = std::chrono::high_resolution_clock::now();
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (DEBUG&rehash) {
        ofstream retrivalFile;
        if(isHopper == 1){
            retrivalFile.open("/scratch/zqu4/SLIDE-MP/dataset/retrival.txt",  std::ios_base::app);
        }else{
            retrivalFile.open("../dataset/retrival.txt",  std::ios_base::app);
        }

        cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;
        retrivalFile << "Epoch: " << epoch << " " << "numBatch: " << numBatch << endl;
        retrivalFile << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;

        retrivalFile.close();

    }

    delete[] avg_retrieval;

    return logloss;
}

void Network::saveWeights(string file)
{
    for (int i=0; i< _numberOfLayers; i++){
        _hiddenlayers[i]->saveWeights(file);
    }
}

Network::~Network() {

    delete[] _sizesOfLayers;
    for (int i=0; i< _numberOfLayers; i++){
        delete _hiddenlayers[i];
    }
    delete[] _hiddenlayers;
    delete[] _layersTypes;
    delete[] _inputIDs;
}
