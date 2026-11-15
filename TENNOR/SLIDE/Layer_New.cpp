#include "Layer_New.h"
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
    _Nodes = new Node *[noOfNodes] ();
    _type = type;
    _noOfActive = floor(_noOfNodes * Sparsity);
    _K = K;
    _L = L;
    _batchsize = batchsize;
    _RangePow = RangePow;
    _previousLayerNumOfNodes = previousLayerNumOfNodes;

    std::cout<<"Start Init Layer "<< layerID <<", Requester"<<std::endl;
    int ReqsPerInput= ( (K * (_K - 1) * (_K - 2) ) / (3*2)) * (2*2*2) + ( ( _K * (_K - 1) ) / (2) ) * (2*2) + _K + _K + 1; // up to 3 hash values, replace with Max-2, and Max-3
    int totalReqs = ReqsPerInput * _batchsize;
    _requester = new Requester(totalReqs);

    int lshsize = 1 << RangePow;
    size_t b;
    if (totalReqs <= lshsize){
        b = totalReqs;
    } else{
        b = lshsize;
    }
    size_t z = _batchsize;
    std::cout<<"Start Init Layer "<< layerID <<", OHT"<<std::endl;
    _oht = new Oht(b, z);
    std::cout<<"End Init Layer "<< layerID <<", OHT"<<std::endl;

// Initialize hash tables and add the nodes (Beidi).
    _hashTables = new LSH(_K, _L, RangePow);

    for (int li=0; li<_L; ++li)
        for (uint32_t j=0; j<(1u<<RangePow); j++)
            assert(_hashTables->_bucket[li][j]._bucID == (int)j);

    _binids = new int[previousLayerNumOfNodes];
    _dwtaHasher = new DensifiedWtaHash(_K * _L, _K, _L, previousLayerNumOfNodes);

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

    Node* tmpnode = nullptr;
    Node* tmpdum = nullptr;

    tmpdum = new Node(previousLayerNumOfNodes, -1, _layerID, type, batchsize, _weights, _bias[0], _adamAvgMom , _adamAvgVel);
    tmpdum->isDummy = 1;

    int cur_buc_count = -1;

    std::cout<<"Start Init Layer "<< layerID <<", Nodes and store into LSH"<<std::endl;
    // 步骤一：分离真实节点
    std::cout << "Step 1: Separating real nodes into _AllNodes and _overflows..." << std::endl;
    for (int i = 0; i < noOfNodes; i++)
    {
        tmpnode = new Node(previousLayerNumOfNodes, i, _layerID, type, batchsize, _weights+previousLayerNumOfNodes*i, _bias[i], _adamAvgMom+previousLayerNumOfNodes*i , _adamAvgVel+previousLayerNumOfNodes*i);
        _Nodes[i] = new Node (*tmpnode);

        cur_buc_count = addtoHashTable(_Nodes[i]);
        if(cur_buc_count < BUCKETSIZE && cur_buc_count >=0) {
            _AllNodes.push_back(*_Nodes[i]);
        }
        if(cur_buc_count >= BUCKETSIZE && cur_buc_count >=0){
            _overflows.push_back(*_Nodes[i]);
        }
        delete tmpnode;
        tmpnode = nullptr;
    }

    std::cout << "Step 1 Done. _AllNodes real nodes: " << _AllNodes.size()
              << ", _overflows real nodes: " << _overflows.size() << std::endl;

    std::cout << "Step 2 : Starting to populate LSH with dummy" << std::endl;

    // LSH工作集的大小，也就是LSH的总容量
    size_t lsh_working_set_size = (1 << RangePow) * BUCKETSIZE;

    size_t num_padded_dummy_for_lsh = 0;

    // LSH中bucket的总数
    size_t num_buckets = (1 << RangePow);

    for (size_t i = 0; i < num_buckets; ++i) {
        Bucket* current_bucket = &(_hashTables->_bucket[0][i]);
        tmpdum->_lshIndex = i;

        for (int j = 0; j < BUCKETSIZE; ++j) {

            // **重要：防止内存泄漏**
            // 如果这个槽位上已经有一个旧的Node对象，我们需要先删除它
            if (current_bucket->arr[j] == nullptr) {

                current_bucket->arr[j] = new Node(*tmpdum);

                num_padded_dummy_for_lsh++;

                _AllNodes.push_back(*tmpdum);
            }else{
                _overflows.push_back(*tmpdum);
            }

        }
    }

    delete tmpdum;
    tmpdum = nullptr;

    std::cout << "LSH population complete. " << num_padded_dummy_for_lsh << " nodes have been placed into the LSH." << std::endl;
    std::cout << "Allnodes size is :" << _AllNodes.size() <<std::endl;
    if (_AllNodes.size() != (1<<RangePow)*BUCKETSIZE){
        std::cout <<"the size is not equal to target size, error !!!!!!!!" << std::endl;
        exit(1);
    }

    // 步骤四：合并 `_overflows` 到 `_AllNodes`
    std::cout << "\nStep 3: Merging _overflows into _AllNodes..." << std::endl;

// 为了避免昂贵的深拷贝，我们使用 std::move_iterator 将 _overflows 的内容“移动”到 _AllNodes 的末尾
// 这比逐个拷贝要快得多
    _AllNodes.insert(
            _AllNodes.end(),
            std::make_move_iterator(_overflows.begin()),
            std::make_move_iterator(_overflows.end())
    );

    _overflows.clear();

    std::cout << "Step 3 Done. Final total _AllNodes size: " << _AllNodes.size() << " , total num of LSH buckets: "<< (L * (1<<RangePow)) << std::endl;
    if (_AllNodes.size() != (noOfNodes + (1<<RangePow)*BUCKETSIZE)){
        std::cout << "num of nodes in _ALLnodes is not equal to target size!!!! " <<std::endl;
        exit(1);
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
        delete _dwtaHasher;
        delete[] _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash(_K * _L, _K, _L, _previousLayerNumOfNodes);
    } else if (HashFunction == 3) {

        delete _MinHasher,  _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash(_K * _L, _previousLayerNumOfNodes);
        _MinHasher->getMap(_previousLayerNumOfNodes, _binids);
    } else if (HashFunction == 4) {
        delete _srp;
        _srp = new SparseRandomProjection(_previousLayerNumOfNodes, _K * _L, Ratio);

    }
}

void Layer::updateRandomNodes()
{
}

int Layer::addtoHashTable(Node *node) {
    int *hashes;
    if(HashFunction==1) {
        hashes = _wtaHasher->getHash(node->_weights);
    }else if (HashFunction==2) {
        hashes = _dwtaHasher->getHashEasy(node->_weights, node->_dim, TOPK);
    }else if (HashFunction==3) {
        hashes = _MinHasher->getHashEasy(_binids, node->_weights, node->_dim, TOPK);
    }else if (HashFunction==4) {
        hashes = _srp->getHash(node->_weights, node->_dim);
    }

    int * hashIndices = _hashTables->hashesToIndex(hashes);
    node->_lshIndex = hashIndices[0];
    int * bucketIndices = _hashTables->add(hashIndices, node);

    int buccount = _hashTables->_bucket[0][hashIndices[0]]._counts;

    delete [] hashes;
    delete [] hashIndices;
    delete [] bucketIndices;

    return buccount;

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

int Layer::ComputeActivation_FirstLayer(int **inputIndices, float **inputValues, int *lengths, int ***activenodesperlayer, float ***activeValuesperlayer, int **sizes, int layerID, int inputID, float Sparsity) {
    int len;
    int in =0;
    int inO = 0;
    int inNew = 0;
    if(Sparsity==1.0){
        len = _noOfNodes;
        sizes[inputID][layerID] = len;
        activenodesperlayer[inputID][layerID] = new int[len]; //assuming not intitialized;
        for (int i = 0; i < len; i++)
        {
            activenodesperlayer[inputID][layerID][i] = i;
        }
    }

    activeValuesperlayer[inputID][layerID] = new float[len]; //assuming its not initialized else memory leak;

    for (int i = 0; i < len; i++)
    {
        activeValuesperlayer[inputID][layerID][i] = _Nodes[i]->getActivation(inputIndices[inputID], inputValues[inputID], lengths[inputID], inputID);
    }

    return len;

}

void Layer::query(int ***activenodesperlayer, float *** activeValuesperlayer, int **sizes, int layerID, int inputID, float Sparsity, int **requestIDs) {
    int *hashes;
    int *hashes2;
    int *hashes3;

    hashes = _dwtaHasher->getHash(activenodesperlayer[inputID][layerID], activeValuesperlayer[inputID][layerID],
                                  sizes[inputID][layerID]);
    hashes2 = _dwtaHasher->getHash2(activenodesperlayer[inputID][layerID], activeValuesperlayer[inputID][layerID],
                                    sizes[inputID][layerID]);
    hashes3 = _dwtaHasher->getHash3(activenodesperlayer[inputID][layerID], activeValuesperlayer[inputID][layerID],
                                    sizes[inputID][layerID]);

    _hashTables->getRequestIDs(hashes, hashes2, hashes3, requestIDs, inputID);
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

    if (_Nodes) {
        for (int i = 0; i < _noOfNodes; ++i) {
            delete _Nodes[i];
        }
        delete[] _Nodes;
        _Nodes = nullptr;
    }

     if (_type == NodeType::Softmax)
    {
        delete[] _normalizationConstants;
        delete[] _inputIDs;
        _normalizationConstants = nullptr;
        _inputIDs = nullptr;
    }

    delete [] _weights;
    delete [] _bias;

    delete[] _adamAvgMom;
    delete[] _adamAvgVel;

    _weights     = nullptr;
    _bias        = nullptr;
    _adamAvgMom  = nullptr;
    _adamAvgVel  = nullptr;

    delete _wtaHasher;
    delete _dwtaHasher;
    delete _srp;
    delete _MinHasher;

     _wtaHasher   = nullptr;
    _dwtaHasher  = nullptr;
    _MinHasher   = nullptr;
    _srp         = nullptr;

    delete _hashTables;
    _hashTables  = nullptr;

    delete[] _binids;
    _binids = nullptr;

    delete _requester;
    delete _oht;
    _requester = nullptr;
    _oht       = nullptr;
}
