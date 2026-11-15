#include "Network.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include "Config.h"
#include <fstream>
#include <omp.h>
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

        //inference
        for (int j = 0; j < _numberOfLayers; j++) {
            int tmp = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], 0, _Sparsity[_numberOfLayers+j], -1, 0, 0);
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


int Network::ProcessInput(int **inputIndices, float **inputValues, int *lengths, int **labels, int *labelsize, int iter, bool rehash, bool rebuild, int numBatch, int epoch, int* L) {


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

    for (int l=0; l<_numberOfLayers ;l++) {
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
            if(isLog == 1){
                LSHfile << "Layer ID : " << _hiddenlayers[l]->_layerID << endl;
            }

        }
        if (tmpRebuild){
            _hiddenlayers[l]->_hashTables->clear();
            _hiddenlayers[l]->updateTable();
        }
        int ratio = 1;
# pragma omp parallel for
        for (int m=0; m< _hiddenlayers[l]->_noOfNodes; m++) {
            Node *tmp = _hiddenlayers[l]->getNodebyID(m);

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
            if (tmpRehash) {

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

                //m is node ID

                delete[] hashes;
                delete[] hashIndices;
                delete[] bucketIndices;
            }
        }
    }

    if(isLog == 1){
        LSHfile.close();
    }

    # pragma omp parallel for
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = lengths[i];
        int in;
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < _numberOfLayers; j++) {
            in = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], labelsize[i], _Sparsity[j], iter*_currentBatchSize+i, tmpRehash, tmpRebuild);
            avg_retrieval[j]+=in;
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        //Now backpropagate.
        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            for (int k = 0; k < sizes[j + 1]; k++) {
                if (j == _numberOfLayers - 1) {
                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->ComputeExtaStatsForSoftMax(
                            _hiddenlayers[j]->getNomalizationConstant(i), i, labels[i], labelsize[i]);
                }

                if (j != 0) {
                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->backPropagate(
                            _hiddenlayers[j - 1]->getAllNodes(), activenodesperlayer[j], sizes[j], tmplr, i);
                } else {

                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->backPropagateFirstLayer(
                            inputIndices[i], inputValues[i], lengths[i], tmplr, i);
                }
            }

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

    if (DEBUG&rehash) {
        ofstream retrivalFile;
        if(isHopper == 1){
            retrivalFile.open(retrivalfile,  std::ios_base::app);
        }else{
            retrivalFile.open("../dataset/retrival.txt",  std::ios_base::app);
        }

        cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;
        retrivalFile << "Epoch: " << epoch << " " << "numBatch: " << numBatch << endl;
        retrivalFile << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;

        retrivalFile.close();

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
