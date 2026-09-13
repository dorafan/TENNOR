#pragma once
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <string>

#include "Config.h"


using namespace std;

enum NodeType
{ ReLU, Softmax};

class Node
{
private:


public:
    float* _lastGradients;
    int* _ActiveinputIds;
    NodeType _type;

    int _previousLayerNumOfNodes;

	float* _lastDeltaforBPs;
    float* _lastActivations;
    int _currentBatchsize;
    int _dim, _layerNum, _IDinLayer;
	int* _indicesInTables;
	int* _indicesInBuckets;
	float* _weights;
	float* _mirrorWeights;
	float* _adamAvgMom;
	float* _adamAvgVel;
	float* _t;
	int* _update;
	float _bias =0;
	float _tbias = 0;
	float _adamAvgMombias=0;
	float _adamAvgVelbias=0;
	float _mirrorbias =0;
	float* getTestActivation();
	float* getLastDeltaForBPs();

    int oram_ID = 0;

	Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, int previousLayerNumOfNodes);
	void updateWeights(float* newWeights, float newbias);
	float getLastActivation(int inputID);
	void incrementDelta(int inputID, float incrementValue);
	float getActivation(int* indices, float* values, int length, int inputID);
	void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize);
	void backPropagate(Node** previousNodes,int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID);
	void backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID);
	~Node();


    std::string serialize() const;
    void deserialize(const std::string& buffer);


	float purturbWeight(int weightid, float delta);
	float getGradient(int weightid, int inputID, float InputVal);
};