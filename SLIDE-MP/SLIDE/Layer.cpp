#include "Layer.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <climits>
#include "Config.h"
#include <bitset>
#include <fstream>

using namespace std;


Layer::Layer(int noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize,  int K, int L, int RangePow, float Sparsity, float* weights, float* bias, float *adamAvgMom, float *adamAvgVel) {
    _layerID = layerID;
    _noOfNodes = noOfNodes;
    _Nodes = new Node *[noOfNodes];
    _type = type;
    _noOfActive = floor(_noOfNodes * Sparsity);
    _K = K;
    _L = L;
    _batchsize = batchsize;
    _RangeRow = RangePow;
    _previousLayerNumOfNodes = previousLayerNumOfNodes;

// create a list of random nodes just in case not enough nodes from hashtable for active nodes.
    _randNode = new int[_noOfNodes];
    for (int n = 0; n < _noOfNodes; n++) {
        _randNode[n] = n;
    }

    std::random_shuffle(_randNode, _randNode + _noOfNodes);

//TODO: Initialize Hash Tables and add the nodes. Done by Beidi
    _hashTables = new LSH(_K, _L, RangePow);

    if (HashFunction == 1) {
        _wtaHasher = new WtaHash(_K * _L, previousLayerNumOfNodes);
    } else if (HashFunction == 2) {
        _binids = new int[previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash(_K * _L, _K, _L, previousLayerNumOfNodes);
    } else if (HashFunction == 3) {
        _binids = new int[previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash(_K * _L, previousLayerNumOfNodes);
        _MinHasher->getMap(previousLayerNumOfNodes, _binids);
    } else if (HashFunction == 4) {
        _srp = new SparseRandomProjection(previousLayerNumOfNodes, _K * _L, Ratio);
    }

    if (LOADWEIGHT) {
        _weights = weights;
        _bias = bias;

        if (ADAM){
            _adamAvgMom = adamAvgMom;
            _adamAvgVel = adamAvgVel;
        }

    }else{
        _weights = new float[_noOfNodes * previousLayerNumOfNodes]();
        _bias = new float[_noOfNodes];
        random_device rd;
        default_random_engine dre(rd());
        normal_distribution<float> distribution(0.0, 0.01);

        generate(_weights, _weights + _noOfNodes * previousLayerNumOfNodes, [&] () { return distribution(dre); });
        generate(_bias, _bias + _noOfNodes, [&] () { return distribution(dre); });


        if (ADAM)
        {
            _adamAvgMom = new float[_noOfNodes * previousLayerNumOfNodes]();
            _adamAvgVel = new float[_noOfNodes * previousLayerNumOfNodes]();

        }
    }
        int total_time = 0;

    auto t1 = std::chrono::high_resolution_clock::now();


    #pragma omp parallel for
    for (size_t i = 0; i < noOfNodes; i++)
    {
        _Nodes[i] = new Node(previousLayerNumOfNodes, i, _layerID, type, batchsize, _weights+previousLayerNumOfNodes*i, _bias[i], _adamAvgMom+previousLayerNumOfNodes*i , _adamAvgVel+previousLayerNumOfNodes*i);
        addtoHashTable(_Nodes[i]->_weights, previousLayerNumOfNodes, _Nodes[i]->_bias, i);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout<< noOfNodes<<" "<<1.0 * timeDiffInMiliseconds<<std::endl;

    if (type == NodeType::Softmax)
    {
        _normalizationConstants = new float[batchsize]();
        _inputIDs = new int[batchsize]();
    }
}


void Layer::updateTable()
{

    if (HashFunction == 1) {
         delete _wtaHasher;
        _wtaHasher = new WtaHash(_K * _L, _previousLayerNumOfNodes);
    } else if (HashFunction == 2) {
         delete _dwtaHasher, _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash(_K * _L, _K, _L, _previousLayerNumOfNodes);
    } else if (HashFunction == 3) {

         delete _MinHasher,  _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash(_K * _L, _previousLayerNumOfNodes);
        _MinHasher->getMap(_previousLayerNumOfNodes, _binids);
    } else if (HashFunction == 4) {

        _srp = new SparseRandomProjection(_previousLayerNumOfNodes, _K * _L, Ratio);

    }
}


void Layer::updateRandomNodes()
{
    std::random_shuffle(_randNode, _randNode + _noOfNodes);
}


void Layer::addtoHashTable(float* weights, int length, float bias, int ID)
{
    int *hashes;
    if(HashFunction==1) {
        hashes = _wtaHasher->getHash(weights);
    }else if (HashFunction==2) {
        hashes = _dwtaHasher->getHashEasy(weights, length, TOPK);
    }else if (HashFunction==3) {
        hashes = _MinHasher->getHashEasy(_binids, weights, length, TOPK);
    }else if (HashFunction==4) {
        hashes = _srp->getHash(weights, length);
    }

    int * hashIndices = _hashTables->hashesToIndex(hashes);
    int * bucketIndices = _hashTables->add(hashIndices, ID+1);

    _Nodes[ID]->_indicesInTables = hashIndices;
    _Nodes[ID]->_indicesInBuckets = bucketIndices;

    delete [] hashes;

}


Node* Layer::getNodebyID(int nodeID)
{
    assert(("nodeID less than _noOfNodes" , nodeID < _noOfNodes));
    return _Nodes[nodeID];
}


Node ** Layer::getAllNodes()
{
    return _Nodes;
}


float Layer::getNomalizationConstant(int inputID)
{
    assert(("Error Call to Normalization Constant for non - softmax layer", _type == NodeType::Softmax));
    return _normalizationConstants[inputID];
}


float innerproduct(int* index1, float* value1, int len1, float* value2){
    float total = 0;
    for (int i=0; i<len1; i++){
        total+=value1[i]*value2[index1[i]];
    }
    return total;
}


float collision(int* hashes, int* table_hashes, int k, int l){
    int cp = 0;
    for (int i=0; i<l; i=i+k){
        int tmp = 0;
        for (int j=i; j< i+k;j++){
            if(hashes[j]==table_hashes[j]){
                tmp++;
            }
        }
        if (tmp==k){
            cp++;
        }
    }
    return cp*1.0/(l/k);
}


int Layer::queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* lengths, int layerIndex, int inputID, int* label, int labelsize, float Sparsity, int iter, bool tmpRehash, bool tmpRebuild)
{
    ofstream mipsFile;
    ofstream signatures;
    if(isLog == 1){
        if(isHopper == 1){
            mipsFile.open("/scratch/zqu4/SLIDE-MP/dataset/mipsfile.txt",  std::ios_base::app);
            signatures.open("/scratch/zqu4/SLIDE-MP/dataset/signatures.txt",  std::ios_base::app);
        }
        else{
            mipsFile.open("../dataset/mipsfile.txt",  std::ios_base::app);
            signatures.open("../dataset/signatures.txt",  std::ios_base::app);
        }
    }

    //Beidi. Query out all the candidate nodes
    int len;
    int in =0;
    int inO = 0;
    int inNew = 0;
    if(Sparsity==1.0){
        len = _noOfNodes;
        lengths[layerIndex + 1] = len;
        activenodesperlayer[layerIndex + 1] = new int[len]; //assuming not intitialized;
        for (int i = 0; i < len; i++)
        {
            activenodesperlayer[layerIndex + 1][i] = i;
        }
    }
    else {


        if (Mode==1) {


            int *hashes;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (HashFunction == 1) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (HashFunction == 2) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            } else if (HashFunction == 3) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], TOPK);
            } else if (HashFunction == 4) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int **actives = _hashTables->retrieveRaw(hashIndices);

            auto t00 = std::chrono::high_resolution_clock::now();

            auto t1 = std::chrono::high_resolution_clock::now();
            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        counts[label[i]] = _L;
                    }
                }
            }


            for (int i = 0; i < _L; i++) {
                if (actives[i] == NULL) {
                    continue;
                } else {
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = actives[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            auto t11 = std::chrono::high_resolution_clock::now();

            //thresholding
            auto t3 = std::chrono::high_resolution_clock::now();
            vector<int> vect;
            for (auto &&x : counts){
                if (x.second>THRESH){
                    vect.push_back(x.first);
                }
            }

            len = vect.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = vect[i];
            }
            auto t33 = std::chrono::high_resolution_clock::now();
            in = len;

            delete[] hashes;
            delete[] hashIndices;
            delete[] actives;

        }
        if (Mode==4) {
            int *hashes;
            int *hashes2;
            int *hashes3;
            if (HashFunction == 1) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (HashFunction == 2) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
                hashes2 = _dwtaHasher->getHash2(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                               lengths[layerIndex]);
                hashes3 = _dwtaHasher->getHash3(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                                lengths[layerIndex]);
            } else if (HashFunction == 3) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], TOPK);
            } else if (HashFunction == 4) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            if(isLog == 1){
                signatures << "iter : " << iter << endl;
                for (int l = 0; l < _L; l++){
                    for (int k = 0; k < _K; k++){
                        signatures<< hashes[l*_K+k] << "\t";
                    }
                }
                signatures << endl;
                for (int l = 0; l < _L; l++){
                    for (int k = 0; k < _K; k++){
                        signatures<< hashes2[l*_K+k] << "\t";
                    }
                }
                signatures << endl;
                for (int l = 0; l < _L; l++){
                    for (int k = 0; k < _K; k++){
                        signatures<< hashes3[l*_K+k] << "\t";
                    }
                }
                signatures << endl;
            }

            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int ***actives = _hashTables->retrieveRawMP(hashes, hashes2);
            int ** activesO = _hashTables->retrieveRaw(hashIndices);
            int bucs = _K * (_K - 1) * (_K - 2) / (3*2) + _K * (_K - 1) / (2) + _K;

            int ***activesNew = _hashTables->retrieveRawMPNew(hashes, hashes2, hashes3);
            int bucsNew = _K * (_K - 1) * (_K - 2) / (3*2) * (2*2*2) + _K * (_K - 1) / (2) * (2*2) + _K + _K;

            std::map<int, size_t> countsNew;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        countsNew[label[i]] = _L;
                    }
                }
            }


            for (int i = 0; i < _L; i++) {
                if (activesNew[i] == NULL) {
                    continue;
                } else {
                    for(int k = 0; k < bucsNew; k++){
                        if(activesNew[i][k] == NULL){
                            continue;
                        }else{
                            for (int j = 0; j < BUCKETSIZE; j++) {
                                int tempID = activesNew[i][k][j] - 1;
                                if (tempID >= 0) {
                                    countsNew[tempID] += 1;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < _L; i++) {
                if (activesO[i] == NULL) {
                    continue;
                } else {
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = activesO[i][j] - 1;
                        if (tempID >= 0) {
                            countsNew[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }

            inNew = countsNew.size();

            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        counts[label[i]] = _L;
                    }
                }
            }


            for (int i = 0; i < _L; i++) {
                if (actives[i] == NULL) {
                    continue;
                } else {
                    for(int k = 0; k < bucs; k++){
                        if(actives[i][k] == NULL){
                            continue;
                        }else{
                            for (int j = 0; j < BUCKETSIZE; j++) {
                                int tempID = actives[i][k][j] - 1;
                                if (tempID >= 0) {
                                    counts[tempID] += 1;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < _L; i++) {
                if (activesO[i] == NULL) {
                    continue;
                } else {
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = activesO[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }

            in = counts.size();

            std::map<int, size_t> countsO;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        countsO[label[i]] = _L;
                    }
                }
            }


            for (int i = 0; i < _L; i++) {
                if (activesO[i] == NULL) {
                    continue;
                } else {
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = activesO[i][j] - 1;
                        if (tempID >= 0) {
                            countsO[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }

            inO = countsO.size();

            if (isBruteForce == 0 && isRandom == 0){
                if (isMPWTA == 1){
                    len = countsNew.size();
                    lengths[layerIndex + 1] = len;
                    activenodesperlayer[layerIndex + 1] = new int[len];

                    int i=0;
                    for (auto &&x : countsNew) {
                        activenodesperlayer[layerIndex + 1][i] = x.first;
                        i++;
                    }
                }
                if (isMPWTA == 0){
                    len = countsO.size();
                    lengths[layerIndex + 1] = len;
                    activenodesperlayer[layerIndex + 1] = new int[len];

                    int i=0;
                    for (auto &&x : countsO) {
                        activenodesperlayer[layerIndex + 1][i] = x.first;
                        i++;
                    }
                }

            }

            if (isBruteForce == 1 && isRandom == 0){
                std::map<int, size_t> countsBF;
                if (_type == NodeType::Softmax) {
                    if (labelsize > 0) {
                        for (int i=0; i<labelsize ;i++){
                            countsBF[label[i]] = _L;
                        }
                    }
                }

                int BFcount = 0;
                if (BFnumber == 0){
                    BFcount = inNew;
                }else{
                    BFcount = BFnumber;
                }

                if (NNSorMPS == 0){
                    vector<float> nns;
                    for(int i=0; i < _noOfNodes; i++){
                        float distance = 0;
                        for (int k = 0; k < lengths[layerIndex]; k++){
                            int indextmp = activenodesperlayer[layerIndex][k];
                            distance += pow((activeValuesperlayer[layerIndex][k] - getNodebyID(i)->_weights[indextmp]), 2);
                        }
                        distance = ::sqrt(distance);
                        nns.push_back(distance);
                    }

                    nth_element(nns.begin(), nns.begin()+BFcount, nns.end(), myfunction2);
                    float nnsflag = nns[BFcount];

                    for(int i=0; i < _noOfNodes; i++){
                        float distance = 0;
                        for (int k = 0; k < lengths[layerIndex]; k++){
                            int indextmp = activenodesperlayer[layerIndex][k];
                            distance += pow((activeValuesperlayer[layerIndex][k] - getNodebyID(i)->_weights[indextmp]), 2);
                        }
                        distance = ::sqrt(distance);

                        if (distance <= nnsflag){
                            countsBF[i] += 1;
                        }

                    }

                    nns.clear();

                }
                else{
                    vector<float> ips;
                    for(int i =0; i < _noOfNodes; i++){
                        float ip = 0;
                        for (int k = 0; k < lengths[layerIndex]; k++){
                            int indextmp = activenodesperlayer[layerIndex][k];
                            ip += activeValuesperlayer[layerIndex][k] * getNodebyID(i)->_weights[indextmp];
                        }
                        ips.push_back(ip);
                    }

                    nth_element(ips.begin(), ips.begin()+BFcount,ips.end(), myfunction);
                    float ipsflag = ips[BFcount];

                    for(int i =0; i < _noOfNodes; i++){
                        float ip = 0;
                        for (int k = 0; k < lengths[layerIndex]; k++){
                            int indextmp = activenodesperlayer[layerIndex][k];
                            ip += activeValuesperlayer[layerIndex][k] * getNodebyID(i)->_weights[indextmp];
                        }

                        if (ip >= ipsflag){
                            countsBF[i] += 1;
                        }
                    }

                    ips.clear();

                }


                len = countsBF.size();
                lengths[layerIndex + 1] = len;
                activenodesperlayer[layerIndex + 1] = new int[len];

                int i=0;
                for (auto &&x : countsBF) {
                    activenodesperlayer[layerIndex + 1][i] = x.first;
                    i++;
                }

                for (auto iter = countsBF.begin(); iter != countsBF.end();){
                    iter = countsBF.erase(iter);
                }

            }

            if (isRandom == 1){
                std::map<int, size_t> countsRand;
                if (_type == NodeType::Softmax) {
                    if (labelsize > 0) {
                        for (int i=0; i<labelsize ;i++){
                            countsRand[label[i]] = _L;
                        }
                    }
                }

                srand(unsigned (time(NULL)));
                while (countsRand.size() < inNew){

                    size_t rnode = rand() % _noOfNodes;
                    if (countsRand.count(rnode) == 0) {
                        countsRand[rnode] += 1;
                    }
                    if (countsRand.size() >= inNew) {
                        break;
                    }
                }

                len = countsRand.size();
                lengths[layerIndex + 1] = len;
                activenodesperlayer[layerIndex + 1] = new int[len];

                int i=0;
                for (auto &&x : countsRand) {
                    activenodesperlayer[layerIndex + 1][i] = x.first;
                    i++;
                }


                for (auto iter = countsRand.begin(); iter != countsRand.end();){
                    iter = countsRand.erase(iter);
                }


            }

            delete[] hashes;
            delete[] hashes2;
            delete[] hashIndices;
            delete[] hashes3;
            delete[] actives;

            delete[] activesO;

            delete[] activesNew;

            for (auto iter = counts.begin(); iter != counts.end();){
                iter = counts.erase(iter);
            }
            for (auto iter = countsO.begin(); iter != countsO.end();){
                iter = countsO.erase(iter);
            }
            for (auto iter = countsNew.begin(); iter != countsNew.end();){
                iter = countsNew.erase(iter);
            }

        }
        else if (Mode == 2 & _type== NodeType::Softmax) {
            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            auto t1 = std::chrono::high_resolution_clock::now();
            bitset <MAPLEN> bs;
            int tmpsize = 0;
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        activenodesperlayer[layerIndex + 1][i] = label[i];
                        bs[label[i]] = 1;
                    }
                    tmpsize = labelsize;
                }
            }

            while(tmpsize<len){
                int v = rand() % _noOfNodes;
                if(bs[v]==0) {
                    activenodesperlayer[layerIndex + 1][tmpsize] = v;
                    bs[v]=1;
                    tmpsize++;
                }
            }



            auto t2 = std::chrono::high_resolution_clock::now();
            auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        }

        else if (Mode==3 & _type== NodeType::Softmax){

            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];
            vector<pair<float, int> > sortW;
            int what = 0;

            for (int s = 0; s < _noOfNodes; s++) {
                float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                         lengths[layerIndex], _Nodes[s]->_weights);
                tmp += _Nodes[s]->_bias;
                if (find(label, label + labelsize, s) != label + labelsize) {
                    sortW.push_back(make_pair(-1000000000, s));
                    what++;
                }
                else{
                    sortW.push_back(make_pair(-tmp, s));
                }
            }

            std::sort(begin(sortW), end(sortW));

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = sortW[i].second;
                if (find (label, label+labelsize, sortW[i].second)!= label+labelsize){
                    in=1;
                }
            }
        }
    }

    activeValuesperlayer[layerIndex + 1] = new float[len]; //assuming its not initialized else memory leak;
    float maxValue = 0;
    if (_type == NodeType::Softmax)
        _normalizationConstants[inputID] = 0;

    int filtered = 0;
    for (int i = 0; i < len; i++)
    {
        activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]]->getActivation(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);
        if(activeValuesperlayer[layerIndex + 1][i]==0){
            filtered++;
        }
        if(_type == NodeType::Softmax){
            if (activeValuesperlayer[layerIndex + 1][i]>maxValue){
                maxValue = activeValuesperlayer[layerIndex + 1][i];
            }
        }
    }

    if(_type == NodeType::Softmax) {
        for (int i = 0; i < len; i++) {
            float realActivation = exp(activeValuesperlayer[layerIndex + 1][i] - maxValue);
            activeValuesperlayer[layerIndex + 1][i] =realActivation;
            _Nodes[activenodesperlayer[layerIndex + 1][i]]->_lastActivations[inputID] = realActivation;
            _normalizationConstants[inputID] += realActivation;
        }
    }

    if(isLog == 1){
        mipsFile.close();
        signatures.close();
    }


    return len;
}


void Layer::saveWeights(string file)
{
    if (_layerID==0) {
        cnpy::npz_save(file, "w_layer_0", _weights, {static_cast<unsigned long>(_noOfNodes), static_cast<unsigned long>(_Nodes[0]->_dim)}, "w");
        cnpy::npz_save(file, "b_layer_0", _bias, {static_cast<unsigned long>(_noOfNodes)}, "a");
        cnpy::npz_save(file, "am_layer_0", _adamAvgMom, {static_cast<unsigned long>(_noOfNodes), static_cast<unsigned long>(_Nodes[0]->_dim)}, "a");
        cnpy::npz_save(file, "av_layer_0", _adamAvgVel, {static_cast<unsigned long>(_noOfNodes), static_cast<unsigned long>(_Nodes[0]->_dim)}, "a");
        cout<<"save for layer 0"<<endl;
        cout<<_weights[0]<<" "<<_weights[1]<<endl;
    }else{
        cnpy::npz_save(file, "w_layer_"+ to_string(_layerID), _weights, {static_cast<unsigned long>(_noOfNodes), static_cast<unsigned long>(_Nodes[0]->_dim)}, "a");
        cnpy::npz_save(file, "b_layer_"+ to_string(_layerID), _bias, {static_cast<unsigned long>(_noOfNodes)}, "a");
        cnpy::npz_save(file, "am_layer_"+ to_string(_layerID), _adamAvgMom, {static_cast<unsigned long>(_noOfNodes), static_cast<unsigned long>(_Nodes[0]->_dim)}, "a");
        cnpy::npz_save(file, "av_layer_"+ to_string(_layerID), _adamAvgVel, {static_cast<unsigned long>(_noOfNodes), static_cast<unsigned long>(_Nodes[0]->_dim)}, "a");
        cout<<"save for layer "<<to_string(_layerID)<<endl;
        cout<<_weights[0]<<" "<<_weights[1]<<endl;
    }
}

bool Layer::myfunction(float i, float j) {
    return (i > j);
}
bool Layer::myfunction2(float i, float j) {
    return (i < j);
}


Layer::~Layer()
{

    for (size_t i = 0; i < _noOfNodes; i++)
    {
        free(_Nodes[i]);
        if (_type == NodeType::Softmax)
        {
            delete[] _normalizationConstants;
            delete[] _inputIDs;
        }
    }
    delete [] _Nodes;
    delete [] _weights;
    delete [] _bias;

    delete _wtaHasher;
    delete _dwtaHasher;
    delete _srp;
    delete _MinHasher;
    delete [] _randNode;
}
