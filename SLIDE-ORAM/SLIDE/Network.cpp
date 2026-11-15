#include "Network.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include "Config.h"
#include <omp.h>

#include <random>
#include <chrono> 

#define DEBUG 1
using namespace std;

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
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], inputdim, i, _layersTypes[i], _currentBatchSize, K[i], L[i], RangePow[i], Sparsity[i], weight, bias, adamAvgMom, adamAvgVel);
        }
    }
    cout << "after layer" << endl;
    _inputIDs = new int[_currentBatchSize]();

    cout << "Serialize Nodes and LSH buckets And write into ORAM" <<endl;
    for (int i = 0; i < _numberOfLayers; i ++) {
        if (Sparsity[i] != 1) {
            if (TEST){
                std::cout << "START Nodes Serialize" << endl;
            }

            for (int o = 0; o < _hiddenlayers[i]->_noOfNodes; o++){
                pathORAM.write(_hiddenlayers[i]->getNodebyID(o)->oram_ID, _hiddenlayers[i]->getNodebyID(o)->serialize());
            }

            if (TEST){
                std::cout << "START Buckets Serialize" << endl;
            }
            for (int o = 0; o < _hiddenlayers[i]->_L; o++) {
                for (int p = 0; p < (1 << _hiddenlayers[i]->_RangeRow); p++) {
                    pathORAM.write(_hiddenlayers[i]->_hashTables->_bucket[o][p].oram_ID, _hiddenlayers[i]->_hashTables->_bucket[o][p].serialize());
                }
            }
        }
    }
    cout << "End Serialize and initial ORAM write" << endl;
}

Layer *Network::getLayer(int LayerID) {
    if (LayerID < _numberOfLayers)
        return _hiddenlayers[LayerID];
    else {
        cout << "LayerID out of bounds" << endl;

    }
}

int Network::predictClass(int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize) {
    int correctPred = 0;
    int inhashtable = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:correctPred)
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = length[i];

        for (int j = 0; j < _numberOfLayers; j++) {
            int tmp = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], 0, _Sparsity[_numberOfLayers+j], -1);
            inhashtable+=tmp;
        }

        //compute softmax
        int noOfClasses = sizes[_numberOfLayers];
        float max_act = -222222222;
        int predict_class = -1;
        for (int k = 0; k < noOfClasses; k++) {
            float cur_act = _hiddenlayers[_numberOfLayers - 1]->getNodebyID(activenodesperlayer[_numberOfLayers][k])->getLastActivation(i);
            if (max_act < cur_act) {
                max_act = cur_act;
                predict_class = activenodesperlayer[_numberOfLayers][k];
            }
        }

        if (std::find (labels[i], labels[i]+labelsize[i], predict_class)!= labels[i]+labelsize[i]) {
            correctPred++;
        }

        delete[] sizes;
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activenodesperlayer[j];
            delete[] activeValuesperlayer[j];
        }
        delete[] activenodesperlayer;
        delete[] activeValuesperlayer;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Inference " <<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;

    return correctPred;
}

int Network::ProcessInput(int **inputIndices, float **inputValues, int *lengths, int **labels, int *labelsize, int iter, bool rehash, bool rebuild) {

    float logloss = 0.0;
    int lowconf = 0;
    float prob = 0.0;
    int* avg_retrieval = new int[_numberOfLayers]();

    int check = 0;
    if(iter%6946==6945 ){

        _hiddenlayers[1]->updateRandomNodes();
    }
    float tmplr = _learningRate;
    if (ADAM) {
        tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) /
                (1 - pow(BETA1, iter + 1));
    }else{

    }

    if (TEST) {
        std::cout << "START Batch Training, ITER: " << iter << endl;
    }

    if (TEST) {
        std::cout << "Update Start" << endl;
    }
    auto tups = std::chrono::high_resolution_clock::now();

        auto t1 = std::chrono::high_resolution_clock::now();
    bool tmpRehash;
    bool tmpRebuild;

    for (int l=0; l<1 ;l++) {
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
            _hiddenlayers[l]->_hashTables->clear();
            if (TEST) {
                std::cout << "Need Rehash" << std::endl;
            }
        }
        if (tmpRebuild){
            _hiddenlayers[l]->updateTable();
        }
        int ratio = 1;

        if (TEST) {
            std::cout << "Update Node Start, layer :" << l << endl;
        }
        for (int m=0; m< _hiddenlayers[l]->_noOfNodes; m++) {
            auto tup1 = std::chrono::high_resolution_clock::now();

            auto tup2 = std::chrono::high_resolution_clock::now();

            Node *tmp = _hiddenlayers[l]->getNodebyID(m);

            auto tup3 = std::chrono::high_resolution_clock::now();

            if(ADAM){

                for (int d=0; d< tmp->_dim;d++){
                    tmp->_adamAvgMom[d] = BETA1 * tmp->_adamAvgMom[d] + (1 - BETA1) * tmp->_t[d];
                    tmp->_adamAvgVel[d] = BETA2 * tmp->_adamAvgVel[d] + (1 - BETA2) * tmp->_t[d] * tmp->_t[d];
                    tmp->_weights[d] += ratio*tmplr * tmp->_adamAvgMom[d] / (sqrt(tmp->_adamAvgVel[d]) + EPS);
                    tmp->_t[d] = 0;
                }

                tmp->_adamAvgMombias = BETA1 * tmp->_adamAvgMombias + (1 - BETA1) * tmp->_tbias;
                tmp->_adamAvgVelbias = BETA2 * tmp->_adamAvgVelbias + (1 - BETA2) * tmp->_tbias * tmp->_tbias;
                tmp->_bias += ratio*tmplr * tmp->_adamAvgMombias / (sqrt(tmp->_adamAvgVelbias) + EPS);
                tmp->_tbias = 0;

            }
            else{
                std::copy(tmp->_mirrorWeights, tmp->_mirrorWeights+(tmp->_dim) , tmp->_weights);
                tmp->_bias = tmp->_mirrorbias;
            }

            auto tup4 = std::chrono::high_resolution_clock::now();

            auto tup5 = std::chrono::high_resolution_clock::now();

            auto tupntimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tup5 - tup1).count();
            updatenodetime += tupntimeDiffInMiliseconds;

        }
    }

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
            _hiddenlayers[l]->_hashTables->clear();
            if (TEST) {
                std::cout << "Need Rehash" << std::endl;
            }
        }
        if (tmpRebuild){
            _hiddenlayers[l]->updateTable();
        }
        int ratio = 1;

        if (TEST) {
            std::cout << "Update Node Start, layer : " << l << endl;
        }
        for (int m=0; m< _hiddenlayers[l]->_noOfNodes; m++) {
            auto tup1 = std::chrono::high_resolution_clock::now();

            int read_id = _hiddenlayers[l]->getNodebyID(m)->oram_ID;
            std::string data = pathORAM.read(read_id);

            auto tup2 = std::chrono::high_resolution_clock::now();

            Node *tmp = _hiddenlayers[l]->getNodebyID(m);

            auto tup3 = std::chrono::high_resolution_clock::now();

            if(ADAM){

                for (int d=0; d< tmp->_dim;d++){
                    tmp->_adamAvgMom[d] = BETA1 * tmp->_adamAvgMom[d] + (1 - BETA1) * tmp->_t[d];
                    tmp->_adamAvgVel[d] = BETA2 * tmp->_adamAvgVel[d] + (1 - BETA2) * tmp->_t[d] * tmp->_t[d];
                    tmp->_weights[d] += ratio*tmplr * tmp->_adamAvgMom[d] / (sqrt(tmp->_adamAvgVel[d]) + EPS);
                    tmp->_t[d] = 0;
                }

                tmp->_adamAvgMombias = BETA1 * tmp->_adamAvgMombias + (1 - BETA1) * tmp->_tbias;
                tmp->_adamAvgVelbias = BETA2 * tmp->_adamAvgVelbias + (1 - BETA2) * tmp->_tbias * tmp->_tbias;
                tmp->_bias += ratio*tmplr * tmp->_adamAvgMombias / (sqrt(tmp->_adamAvgVelbias) + EPS);
                tmp->_tbias = 0;

            }
            else{
                std::copy(tmp->_mirrorWeights, tmp->_mirrorWeights+(tmp->_dim) , tmp->_weights);
                tmp->_bias = tmp->_mirrorbias;
            }

            auto tup4 = std::chrono::high_resolution_clock::now();

            pathORAM.write(read_id, tmp->serialize());

            auto tup5 = std::chrono::high_resolution_clock::now();

            auto tupntimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tup5 - tup1).count();
            updatenodetime += tupntimeDiffInMiliseconds;
            auto tupnoramtimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tup5 - tup4 + tup2 - tup1).count();
            updateNodeOramTime += tupnoramtimeDiffInMiliseconds;

            if (tmpRehash) {

                auto tup6 = std::chrono::high_resolution_clock::now();

                if (TEST) {
                    if (m % 10000 == 0) {
                        std::cout << "LSH update Start, node id : " << m << endl;
                    }

                }

                int *hashes;
                if(HashFunction==1) {
                    hashes = _hiddenlayers[l]->_wtaHasher->getHash(tmp->_weights);
                }else if (HashFunction==2){
                    hashes = _hiddenlayers[l]->_dwtaHasher->getHashEasy(tmp->_weights, tmp->_dim, TOPK);
                }else if (HashFunction==3){
                    hashes = _hiddenlayers[l]->_MinHasher->getHashEasy(_hiddenlayers[l]->_binids, tmp->_weights, tmp->_dim, TOPK);
                }else if (HashFunction==4){
                    hashes = _hiddenlayers[l]->_srp->getHash(tmp->_weights, tmp->_dim);
                }

                int *hashIndices = _hiddenlayers[l]->_hashTables->hashesToIndex(hashes);
                int * bucketIndices = _hiddenlayers[l]->_hashTables->add(hashIndices, m+1);

                auto tup7 = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < _hiddenlayers[l]->_L; i++)
                {
                    int ht_read_id = _hiddenlayers[l]->_hashTables->_bucket[i][hashIndices[i]].oram_ID;
                    std::string data = pathORAM.read(ht_read_id);

                    pathORAM.write(ht_read_id, _hiddenlayers[l]->_hashTables->_bucket[i][hashIndices[i]].serialize());
                }

                auto tup8 = std::chrono::high_resolution_clock::now();

                auto tupltimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tup8 - tup6).count();
                rehashtime += tupltimeDiffInMiliseconds;
                auto tuploramtimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tup8 - tup7).count();
                rehashLshOramTime += tuploramtimeDiffInMiliseconds;

                delete[] hashes;
                delete[] hashIndices;
                delete[] bucketIndices;
            }
        }
    }

    auto tupe = std::chrono::high_resolution_clock::now();
    int tuptimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tupe - tups).count();
    updatetime += tuptimeDiffInMiliseconds;

    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = lengths[i];

        int in;

        if (TEST) {
            std::cout << "FeedForward Start" << endl;
        }

        auto tffs = std::chrono::high_resolution_clock::now();

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < _numberOfLayers; j++) {
            in = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], labelsize[i], _Sparsity[j], iter*_currentBatchSize+i);
            avg_retrieval[j]+=in;

//            simulate the oram
            if (j != 0) {

                auto tons = std::chrono::high_resolution_clock::now();

                // Pad node accesses to BUCKETSIZE * L with dummy accesses.

                for (int sims = 0; sims < (BUCKETSIZE*_hiddenlayers[j]->_L) ; sims ++) {

                    if (sims < sizes[j+1]){
                         int id_read = _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][sims])->oram_ID;

                        std::string data = pathORAM.read(id_read);

                        pathORAM.write(id_read, _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][sims])->serialize());

                    }else{
                         std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
                         std::uniform_int_distribution<> distr(1, _hiddenlayers[j]->_noOfNodes-2);
                         int randnum = distr(gen);
                         int id_read = _hiddenlayers[j]->getNodebyID(randnum)->oram_ID;
                        std::string data = pathORAM.read(id_read);
                        pathORAM.write(id_read, _hiddenlayers[j]->getNodebyID(randnum)->serialize());
                    }

                }

                auto tone = std::chrono::high_resolution_clock::now();
                auto tontimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tone - tons).count();
                oramnodetime += tontimeDiffInMiliseconds;

                auto tols = std::chrono::high_resolution_clock::now();

                for (int simsht = 0; simsht < _hiddenlayers[j]->_L ; simsht ++) {
                    int id_read = _hiddenlayers[j]->_hashTables->_bucket[simsht][simsht].oram_ID;
                    std::string data = pathORAM.read(id_read);

                    pathORAM.write(id_read, _hiddenlayers[j]->_hashTables->_bucket[simsht][simsht].serialize());

                }

                auto tole = std::chrono::high_resolution_clock::now();
                auto toltimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tole - tols).count();
                oramlshtime += toltimeDiffInMiliseconds;
            }

        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        auto tffe = std::chrono::high_resolution_clock::now();
        auto tfftimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tffe - tffs).count();
        feedforwardtime += tfftimeDiffInMiliseconds;

        if (TEST) {
            std::cout << "BackPropagation Start" << endl;
        }

        auto tbps = std::chrono::high_resolution_clock::now();

        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            if (j == _numberOfLayers - 1){
                for (int k = 0; k < (BUCKETSIZE * _hiddenlayers[j]->_L - sizes[j+1]); k++){
                    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
                    std::uniform_int_distribution<> distr(1, _hiddenlayers[j]->_noOfNodes-2);
                    int randnum = distr(gen);
                    int id_read = _hiddenlayers[j]->getNodebyID(randnum)->oram_ID;
                    std::string data = pathORAM.read(id_read);
                    pathORAM.write(id_read, _hiddenlayers[j]->getNodebyID(randnum)->serialize());
                }
            }
            if (j != 0){
                for (int k = 0; k < (BUCKETSIZE * _hiddenlayers[j]->_L - sizes[j+1]); k++){
                    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
                    std::uniform_int_distribution<> distr(1, _hiddenlayers[j]->_noOfNodes-2);
                    int randnum = distr(gen);
                    int id_read = _hiddenlayers[j]->getNodebyID(randnum)->oram_ID;
                    std::string data = pathORAM.read(id_read);
                    pathORAM.write(id_read, _hiddenlayers[j]->getNodebyID(randnum)->serialize());
                }
            }
            for (int k = 0; k < sizes[j + 1]; k++) {
                if (j == _numberOfLayers - 1) {

                    auto tbp1 = std::chrono::high_resolution_clock::now();

                    int id_read =  _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->oram_ID;
                    std::string data = pathORAM.read(id_read);

                    auto tbp2 = std::chrono::high_resolution_clock::now();

                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->ComputeExtaStatsForSoftMax(
                            _hiddenlayers[j]->getNomalizationConstant(i), i, labels[i], labelsize[i]);

                    auto tbp3 = std::chrono::high_resolution_clock::now();

                    pathORAM.write(id_read, _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->serialize());

                    auto tbp4 = std::chrono::high_resolution_clock::now();

                    auto tbptmptimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tbp4 - tbp3 + tbp2 - tbp1).count();
                    oramnodetime += tbptmptimeDiffInMiliseconds;
                }

                if (j != 0) {
                    auto tbp1 = std::chrono::high_resolution_clock::now();
                    int id_read =  _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->oram_ID;
                    std::string data = pathORAM.read(id_read);

                    auto tbp2 = std::chrono::high_resolution_clock::now();

                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->backPropagate(
                            _hiddenlayers[j - 1]->getAllNodes(), activenodesperlayer[j], sizes[j], tmplr, i);

                    auto tbp3 = std::chrono::high_resolution_clock::now();

                    pathORAM.write(id_read, _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->serialize());

                    auto tbp4 = std::chrono::high_resolution_clock::now();
                    auto tbptmptimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tbp4 - tbp3 + tbp2 - tbp1).count();
                    oramnodetime += tbptmptimeDiffInMiliseconds;
                } else {

                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->backPropagateFirstLayer(
                            inputIndices[i], inputValues[i], lengths[i], tmplr, i);
                }
            }

        }

        auto tbpe = std::chrono::high_resolution_clock::now();
        auto tbptimeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(tbpe - tbps).count();
        backpropagationtime += tbptimeDiffInMiliseconds;

        if (TEST) {
            std::cout << "Free Memory " << endl;
        }

        delete[] sizes;
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activenodesperlayer[j];
            delete[] activeValuesperlayer[j];
        }
        delete[] activenodesperlayer;
        delete[] activeValuesperlayer;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Layer " <<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;

    if (DEBUG&rehash) {

        cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;

    }
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
