#include "Node.h"
#include <random>
#include <math.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

#include <cstring>
#include "Config.h"

using namespace std;

Node::Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, int previousLayerNumOfNodes)
{
	_dim = dim;
	_IDinLayer = nodeID;
	_type = type;
	_layerNum = layerID;
    _currentBatchsize = batchsize;

    _previousLayerNumOfNodes = previousLayerNumOfNodes;

	if (ADAM)
	{
		_adamAvgMom = adamAvgMom;
		_adamAvgVel = adamAvgVel;
		_t = new float[_dim]();

	}

	_lastActivations = new float[_currentBatchsize]();
	_lastDeltaforBPs = new float[_currentBatchsize]();
	_lastGradients = new float[_currentBatchsize]();
	_ActiveinputIds = new int[_currentBatchsize]();

    _weights = weights;
    _bias = bias;
	_mirrorbias = _bias;

    _mirrorWeights = new float[_previousLayerNumOfNodes] ();

}

float Node::getLastActivation(int inputID)
{
	if(_ActiveinputIds[inputID] != 1)
		return 0.0;
	return _lastActivations[inputID];
}

void Node::incrementDelta(int inputID, float incrementValue)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));
	if (_lastActivations[inputID]>0)
		_lastDeltaforBPs[inputID] += incrementValue;
}

float Node::getActivation(int* indices, float* values, int length, int inputID)
{
    if (inputID>_currentBatchsize){

    }
	assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));

	_ActiveinputIds[inputID] = 1; 

	_lastActivations[inputID] = 0;
	for (int i = 0; i < length; i++)
	{
		_lastActivations[inputID] += _weights[indices[i]] * values[i];

	}
	_lastActivations[inputID] += _bias;

	switch (_type)
	{
	case NodeType::ReLU:
		if (_lastActivations[inputID] < 0) {
            _lastActivations[inputID] = 0;
            _lastGradients[inputID] = 1;
			_lastDeltaforBPs[inputID] = 0;

        }else{
            _lastGradients[inputID] = 0;
		}
		break;
	case NodeType::Softmax:

		break;
	default:
		cout << "Invalid Node type from Constructor" <<endl;
		break;
	}

	return _lastActivations[inputID];
}

void Node::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize)
{
	if (_ActiveinputIds[inputID] !=1){

	}
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] ==1));

	_lastActivations[inputID] /= normalizationConstant + 0.0000001;

	_lastGradients[inputID] = 1;
	if (find (label, label+labelsize, _IDinLayer)!= label+labelsize) {
		_lastDeltaforBPs[inputID] = (1.0/labelsize - _lastActivations[inputID]) / _currentBatchsize;
	}
	else {
		_lastDeltaforBPs[inputID] = (-_lastActivations[inputID]) / _currentBatchsize;
	}
}

void Node::backPropagate(Node** previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));

	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		//UpdateDelta before updating weights
		previousNodes[previousLayerActiveNodeIds[i]]->incrementDelta(inputID, _lastDeltaforBPs[inputID] * _weights[previousLayerActiveNodeIds[i]]);
		float grad_t = _lastDeltaforBPs[inputID] * previousNodes[previousLayerActiveNodeIds[i]]->getLastActivation(inputID);
		float grad_tsq = grad_t * grad_t;

		if (ADAM)
		{
			_t[previousLayerActiveNodeIds[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		float biasgrad_t = _lastDeltaforBPs[inputID];
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
		{
			_mirrorbias += learningRate * _lastDeltaforBPs[inputID];
		}

	_ActiveinputIds[inputID] = 0;
	_lastDeltaforBPs[inputID] = 0;
	_lastActivations[inputID] = 0;

}

void Node::backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));

	for (int i = 0; i < nnzSize; i++)
	{
		float grad_t = _lastDeltaforBPs[inputID] * nnzvalues[i];
		float grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_t[nnzindices[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[nnzindices[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		float biasgrad_t = _lastDeltaforBPs[inputID];
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}

	else {
		_mirrorbias += learningRate * _lastDeltaforBPs[inputID];
	}

	_ActiveinputIds[inputID] = 0;
	_lastDeltaforBPs[inputID] = 0;
	_lastActivations[inputID] = 0;
}

Node::~Node()
{

	delete[] _indicesInTables;
	delete[] _indicesInBuckets;
	delete[] _lastActivations;
	delete[] _lastDeltaforBPs;
	delete[] _lastGradients;
	delete[] _ActiveinputIds;

    delete[] _mirrorWeights;

	if (ADAM)
	{
		delete[] _adamAvgMom;
		delete[] _adamAvgVel;
		delete[] _t;
	}
}

// for debugging gradients.
float Node::purturbWeight(int weightid, float delta)
{
	_weights[weightid] += delta;
	return _weights[weightid];
}

float Node::getGradient(int weightid, int inputID, float InputVal)
{
	return -_lastDeltaforBPs[inputID] * InputVal;
}

float* Node::getTestActivation() {
	return _lastActivations;
}

float* Node::getLastDeltaForBPs() {
	return _lastDeltaforBPs;

}

std::string Node::serialize() const {

    size_t size = 0;

    // Fixed-size members
    size += sizeof(int);   // _type
    size += sizeof(int);   // _previousLayerNumOfNodes
    size += sizeof(int);   // _currentBatchsize
    size += sizeof(int);   // _dim
    size += sizeof(int);   // _layerNum
    size += sizeof(int);   // _IDinLayer
    size += sizeof(float); // _bias
    size += sizeof(float); // _tbias
    size += sizeof(float); // _adamAvgMombias
    size += sizeof(float); // _adamAvgVelbias
    size += sizeof(float); // _mirrorbias

    // Arrays
    size += sizeof(float) * _currentBatchsize; // _lastActivations
    size += sizeof(float) * _currentBatchsize; // _lastDeltaforBPs
    size += sizeof(float) * _currentBatchsize; // _lastGradients
    size += sizeof(int) * _currentBatchsize;   // _ActiveinputIds

    size += sizeof(float) * _previousLayerNumOfNodes; // _weights
    size += sizeof(float) * _previousLayerNumOfNodes; // _mirrorWeights

    // Include ADAM arrays if ADAM is enabled
    bool has_adam = ADAM;
    size += sizeof(bool);  // has_adam

    if (has_adam) {
        size += sizeof(float) * _previousLayerNumOfNodes; // _adamAvgMom
        size += sizeof(float) * _previousLayerNumOfNodes; // _adamAvgVel
        size += sizeof(float) * _dim;                     // _t
    }

    std::string buffer(SERIAL_LEN, '\0');
    size_t offset = 0;

    // Serialize fixed-size members
    int type_int = static_cast<int>(_type);
    std::memcpy(&buffer[offset], &type_int, sizeof(int));
    offset += sizeof(int);

    std::memcpy(&buffer[offset], &_previousLayerNumOfNodes, sizeof(int));
    offset += sizeof(int);

    std::memcpy(&buffer[offset], &_currentBatchsize, sizeof(int));
    offset += sizeof(int);

    std::memcpy(&buffer[offset], &_dim, sizeof(int));
    offset += sizeof(int);

    std::memcpy(&buffer[offset], &_layerNum, sizeof(int));
    offset += sizeof(int);

    std::memcpy(&buffer[offset], &_IDinLayer, sizeof(int));
    offset += sizeof(int);

    std::memcpy(&buffer[offset], &_bias, sizeof(float));
    offset += sizeof(float);

    std::memcpy(&buffer[offset], &_tbias, sizeof(float));
    offset += sizeof(float);

    std::memcpy(&buffer[offset], &_adamAvgMombias, sizeof(float));
    offset += sizeof(float);

    std::memcpy(&buffer[offset], &_adamAvgVelbias, sizeof(float));
    offset += sizeof(float);

    std::memcpy(&buffer[offset], &_mirrorbias, sizeof(float));
    offset += sizeof(float);

    // Serialize arrays
    std::memcpy(&buffer[offset], _lastActivations, sizeof(float) * _currentBatchsize);
    offset += sizeof(float) * _currentBatchsize;

    std::memcpy(&buffer[offset], _lastDeltaforBPs, sizeof(float) * _currentBatchsize);
    offset += sizeof(float) * _currentBatchsize;

    std::memcpy(&buffer[offset], _lastGradients, sizeof(float) * _currentBatchsize);
    offset += sizeof(float) * _currentBatchsize;

    std::memcpy(&buffer[offset], _ActiveinputIds, sizeof(int) * _currentBatchsize);
    offset += sizeof(int) * _currentBatchsize;

    std::memcpy(&buffer[offset], _weights, sizeof(float) * _previousLayerNumOfNodes);
    offset += sizeof(float) * _previousLayerNumOfNodes;

    std::memcpy(&buffer[offset], _mirrorWeights, sizeof(float) * _previousLayerNumOfNodes);
    offset += sizeof(float) * _previousLayerNumOfNodes;

    // Serialize has_adam flag
    std::memcpy(&buffer[offset], &has_adam, sizeof(bool));
    offset += sizeof(bool);

    // Serialize ADAM arrays if applicable
    if (has_adam) {
        std::memcpy(&buffer[offset], _adamAvgMom, sizeof(float) * _previousLayerNumOfNodes);
        offset += sizeof(float) * _previousLayerNumOfNodes;

        std::memcpy(&buffer[offset], _adamAvgVel, sizeof(float) * _previousLayerNumOfNodes);
        offset += sizeof(float) * _previousLayerNumOfNodes;

        std::memcpy(&buffer[offset], _t, sizeof(float) * _dim);
        offset += sizeof(float) * _dim;
    }

    return buffer;
}

void Node::deserialize(const std::string& buffer) {

    // Ensure the buffer is at least as big as the data we need to read

    size_t size = 0;

    // Fixed-size members
    size += sizeof(int);   // _type
    size += sizeof(int);   // _previousLayerNumOfNodes
    size += sizeof(int);   // _currentBatchsize
    size += sizeof(int);   // _dim
    size += sizeof(int);   // _layerNum
    size += sizeof(int);   // _IDinLayer
    size += sizeof(float); // _bias
    size += sizeof(float); // _tbias
    size += sizeof(float); // _adamAvgMombias
    size += sizeof(float); // _adamAvgVelbias
    size += sizeof(float); // _mirrorbias

    // Arrays
    size += sizeof(float) * _currentBatchsize; // _lastActivations
    size += sizeof(float) * _currentBatchsize; // _lastDeltaforBPs
    size += sizeof(float) * _currentBatchsize; // _lastGradients
    size += sizeof(int) * _currentBatchsize;   // _ActiveinputIds

    size += sizeof(float) * _previousLayerNumOfNodes; // _weights
    size += sizeof(float) * _previousLayerNumOfNodes; // _mirrorWeights

    // Include ADAM arrays if ADAM is enabled
    bool has_adam = ADAM;
    size += sizeof(bool);  // has_adam

    if (has_adam) {
        size += sizeof(float) * _previousLayerNumOfNodes; // _adamAvgMom
        size += sizeof(float) * _previousLayerNumOfNodes; // _adamAvgVel
        size += sizeof(float) * _dim;                     // _t
    }

    if (buffer.size() < size) {
        if (TEST) {
            std::cout << "NODE Expect Buffre Size : " << size << ",  real buffer size : " << buffer.size() << std::endl;
        }

    }

    size_t offset = 0;

    // Deserialize fixed-size members
    int type_int;
    std::memcpy(&type_int, &buffer[offset], sizeof(int));
    _type = static_cast<NodeType>(type_int);
    offset += sizeof(int);

    std::memcpy(&_previousLayerNumOfNodes, &buffer[offset], sizeof(int));
    offset += sizeof(int);

    std::memcpy(&_currentBatchsize, &buffer[offset], sizeof(int));
    offset += sizeof(int);

    std::memcpy(&_dim, &buffer[offset], sizeof(int));
    offset += sizeof(int);

    std::memcpy(&_layerNum, &buffer[offset], sizeof(int));
    offset += sizeof(int);

    std::memcpy(&_IDinLayer, &buffer[offset], sizeof(int));
    offset += sizeof(int);

    std::memcpy(&_bias, &buffer[offset], sizeof(float));
    offset += sizeof(float);

    std::memcpy(&_tbias, &buffer[offset], sizeof(float));
    offset += sizeof(float);

    std::memcpy(&_adamAvgMombias, &buffer[offset], sizeof(float));
    offset += sizeof(float);

    std::memcpy(&_adamAvgVelbias, &buffer[offset], sizeof(float));
    offset += sizeof(float);

    std::memcpy(&_mirrorbias, &buffer[offset], sizeof(float));
    offset += sizeof(float);

    // Allocate or reallocate arrays
    delete[] _lastActivations;
    _lastActivations = new float[_currentBatchsize];
    std::memcpy(_lastActivations, &buffer[offset], sizeof(float) * _currentBatchsize);
    offset += sizeof(float) * _currentBatchsize;

    delete[] _lastDeltaforBPs;
    _lastDeltaforBPs = new float[_currentBatchsize];
    std::memcpy(_lastDeltaforBPs, &buffer[offset], sizeof(float) * _currentBatchsize);
    offset += sizeof(float) * _currentBatchsize;

    delete[] _lastGradients;
    _lastGradients = new float[_currentBatchsize];
    std::memcpy(_lastGradients, &buffer[offset], sizeof(float) * _currentBatchsize);
    offset += sizeof(float) * _currentBatchsize;

    delete[] _ActiveinputIds;
    _ActiveinputIds = new int[_currentBatchsize];
    std::memcpy(_ActiveinputIds, &buffer[offset], sizeof(int) * _currentBatchsize);
    offset += sizeof(int) * _currentBatchsize;

    // Deserialize _weights and _mirrorWeights

    std::memcpy(_weights, &buffer[offset], sizeof(float) * _previousLayerNumOfNodes);
    offset += sizeof(float) * _previousLayerNumOfNodes;

    delete[] _mirrorWeights;
    _mirrorWeights = new float[_previousLayerNumOfNodes];
    std::memcpy(_mirrorWeights, &buffer[offset], sizeof(float) * _previousLayerNumOfNodes);
    offset += sizeof(float) * _previousLayerNumOfNodes;

    // Deserialize has_adam flag

    std::memcpy(&has_adam, &buffer[offset], sizeof(bool));
    offset += sizeof(bool);

    // Deserialize ADAM arrays if applicable
    if (ADAM) {

        std::memcpy(_adamAvgMom, &buffer[offset], sizeof(float) * _previousLayerNumOfNodes);
        offset += sizeof(float) * _previousLayerNumOfNodes;

        std::memcpy(_adamAvgVel, &buffer[offset], sizeof(float) * _previousLayerNumOfNodes);
        offset += sizeof(float) * _previousLayerNumOfNodes;

        delete[] _t;
        _t = new float[_dim];
        std::memcpy(_t, &buffer[offset], sizeof(float) * _dim);
        offset += sizeof(float) * _dim;
    } else {
        if (_adamAvgMom) { delete[] _adamAvgMom; _adamAvgMom = nullptr; }
        if (_adamAvgVel) { delete[] _adamAvgVel; _adamAvgVel = nullptr; }
        if (_t) { delete[] _t; _t = nullptr; }
    }

}

