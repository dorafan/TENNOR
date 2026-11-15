#pragma once
#include <stdlib.h>
#include <assert.h>
#include <cmath>

#include "obl.h"

using namespace std;

enum NodeType
{ ReLU, Softmax};

// 只打包需要“条件拷贝”的标量字段（原来用 OblChoose 的那几个）
struct NodeScalarBlock {
    int      _IDinLayer;
    NodeType _type;
    int      _layerNum;

    float    _bias = 0;
    float    _mirrorbias = 0;
    float    _tbias = 0;
    float    _adamAvgMombias = 0;
    float    _adamAvgVelbias = 0;

    int      _lshIndex = -1;
    int      _isOverflow = 0;
    int      isDummy = 0;
};

class Node  : public NodeScalarBlock
{
private:
    
public:
    int owns_weights = 1;

    float* _lastDeltaforBPs = nullptr;
    float* _lastActivations = nullptr;

    int _currentBatchsize;
    int _dim;

    float* _weights = nullptr;
    float* _mirrorWeights = nullptr;
    float* _adamAvgMom = nullptr;
    float* _adamAvgVel = nullptr;
    float* _t = nullptr; //for adam
    int* _update = nullptr;

    float* _lastGradients = nullptr;
    int* _ActiveinputIds = nullptr;

    float* getTestActivation();
    float* getLastDeltaForBPs();

    Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel);
    void updateWeights(float* newWeights, float newbias);
    float getLastActivation(int inputID);
    void incrementDelta(int inputID, float incrementValue);
    float getActivation(int* indices, float* values, int length, int inputID);
    void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize);
    void backPropagate(Node** previousNodes,int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID);
    void backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID);
    ~Node();

    //only for debugging
    float purturbWeight(int weightid, float delta);
    float getGradient(int weightid, int inputID, float InputVal);

    void backPropageSingle(Node* previousNode, float learningRate, int inputID);

    void backPropageBias(float learningRate, int inputID);

    Node(int dim, NodeType type, int layerID, int batchsize);

    Node();

    static void obl_copy_node(bool pred, Node* sourcenode, Node* destnode);

    Node(const Node& other);

    Node& operator=(const Node& other);

    Node(Node&& other) noexcept;

    Node& operator=(Node&& other) noexcept;

    void clearForInput(int inputID) {
        _ActiveinputIds[inputID] = 0;
        _lastDeltaforBPs[inputID] = 0;
        _lastActivations[inputID] = 0;
    }

};