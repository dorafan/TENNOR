#include "Node.h"
#include <random>
#include <math.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include "Config.h"

#include "obl.h"
#include "obl_primitives.h"                                             

#include <type_traits>        

using namespace std;

Node::Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel)
{
    owns_weights = 0;
	_dim = dim;
	_IDinLayer = nodeID;
	_type = type;
	_layerNum = layerID;
    _currentBatchsize = batchsize;

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

}

Node::Node() {
    owns_weights = 1;
    _dim = 0;
    _IDinLayer = -1;               // Default node ID in layer, DUMMY(or a random temp node)
    _type = Softmax;
    _layerNum = -1;
    _currentBatchsize = 0;

    _bias = 0.0f;
    _mirrorbias = _bias;

    if (ADAM)
    {
        _adamAvgMom = nullptr;
        _adamAvgVel = nullptr;
        _t = nullptr;
    }
    else
    {
        _adamAvgMom = nullptr;
        _adamAvgVel = nullptr;
        _t = nullptr;
    }

    _lastActivations = nullptr;
    _lastDeltaforBPs = nullptr;
    _lastGradients = nullptr;
    _ActiveinputIds = nullptr;

    _weights = nullptr;
}

//used to generate a dummy neuron with default values,
// also can be used to generate a tmp neuron for oblivious choose and copy (combine with a true node to hide if the true one is selected and copied, or not)
Node::Node(int dim, NodeType type, int layerID, int batchsize)
{
    owns_weights = 1;

    _dim = dim;
    _IDinLayer = -1;               // Default node ID in layer, DUMMY(or a random temp node)
    _type = type;
    _layerNum = layerID;
    _currentBatchsize = batchsize;

    _bias = 0.0f;
    _mirrorbias = _bias;

    if (ADAM)
    {
        _adamAvgMom = new float[_dim]();
        _adamAvgVel = new float[_dim]();
        _t = new float[_dim]();
    }
    else
    {
        _adamAvgMom = nullptr;
        _adamAvgVel = nullptr;
        _t = nullptr;
    }

    _lastActivations = new float[_currentBatchsize]();
    _lastDeltaforBPs = new float[_currentBatchsize]();
    _lastGradients = new float[_currentBatchsize]();
    _ActiveinputIds = new int[_currentBatchsize]();

    _weights = new float[_dim]();
}

void Node::obl_copy_node(bool pred, Node* sourceNode, Node* destNode)
{
    if (!sourceNode || !destNode) {
        return;
    }

    // 这些结构参数我们当作 public，用来决定是否需要重新分配
    const int srcDim = sourceNode->_dim;
    const int srcB   = sourceNode->_currentBatchsize;
    const int dstDim = destNode->_dim;
    const int dstB   = destNode->_currentBatchsize;

    using snoopy::obl::ObliviousBytesAssign;

    auto* srcS = static_cast<const NodeScalarBlock*>(sourceNode);
    auto* dstS = static_cast<NodeScalarBlock*>(destNode);

    // 关键：in-place 写回 dstS（无 outBlock、无逐字段回填）
    ObliviousBytesAssign(pred,
                        sizeof(NodeScalarBlock),
                        static_cast<const void*>(srcS),
                        static_cast<const void*>(dstS),
                        static_cast<void*>(dstS));

    // 先确保 dest 的数组尺寸对得上（这一步只依赖 public 信息）

    if (dstDim != srcDim) {
        delete[] destNode->_weights;
        destNode->_weights = nullptr;

        if (srcDim > 0) {
            destNode->_weights = new float[srcDim]();  // 初始为 0，pred=0 时保持 0
        }
        destNode->_dim = srcDim;
    }

    if (ADAM) {
        // 如果 dim 变了，或者之前是 nullptr，就按 srcDim 重新分配
        if (destNode->_adamAvgMom == nullptr || dstDim != srcDim) {
            delete[] destNode->_adamAvgMom;
            destNode->_adamAvgMom = (srcDim > 0) ? new float[srcDim]() : nullptr;
        }
        if (destNode->_adamAvgVel == nullptr || dstDim != srcDim) {
            delete[] destNode->_adamAvgVel;
            destNode->_adamAvgVel = (srcDim > 0) ? new float[srcDim]() : nullptr;
        }
        if (destNode->_t == nullptr || dstDim != srcDim) {
            delete[] destNode->_t;
            destNode->_t = (srcDim > 0) ? new float[srcDim]() : nullptr;
        }
    }

    if (dstB != srcB) {
        delete[] destNode->_lastActivations;
        delete[] destNode->_lastDeltaforBPs;
        delete[] destNode->_lastGradients;
        delete[] destNode->_ActiveinputIds;

        destNode->_lastActivations = (srcB > 0) ? new float[srcB]() : nullptr;
        destNode->_lastDeltaforBPs = (srcB > 0) ? new float[srcB]() : nullptr;
        destNode->_lastGradients   = (srcB > 0) ? new float[srcB]() : nullptr;
        destNode->_ActiveinputIds  = (srcB > 0) ? new int[srcB]()   : nullptr;

        destNode->_currentBatchsize = srcB;
    }

    // 用 ObliviousBytesAssign 对整段数组做条件拷贝

    const size_t dim_bytes   = static_cast<size_t>(srcDim) * sizeof(float);
    const size_t bs_float_b  = static_cast<size_t>(srcB)   * sizeof(float);
    const size_t bs_int_b    = static_cast<size_t>(srcB)   * sizeof(int);

    if (srcDim > 0 && sourceNode->_weights && destNode->_weights) {
        ObliviousBytesAssign(pred,
                             dim_bytes,
                             static_cast<const void*>(sourceNode->_weights),
                             static_cast<const void*>(destNode->_weights),
                             static_cast<void*>(destNode->_weights));
    }

    if (ADAM && srcDim > 0) {
        if (sourceNode->_adamAvgMom && destNode->_adamAvgMom) {
            ObliviousBytesAssign(pred,
                                 dim_bytes,
                                 static_cast<const void*>(sourceNode->_adamAvgMom),
                                 static_cast<const void*>(destNode->_adamAvgMom),
                                 static_cast<void*>(destNode->_adamAvgMom));
        }
        if (sourceNode->_adamAvgVel && destNode->_adamAvgVel) {
            ObliviousBytesAssign(pred,
                                 dim_bytes,
                                 static_cast<const void*>(sourceNode->_adamAvgVel),
                                 static_cast<const void*>(destNode->_adamAvgVel),
                                 static_cast<void*>(destNode->_adamAvgVel));
        }
        if (sourceNode->_t && destNode->_t) {
            ObliviousBytesAssign(pred,
                                 dim_bytes,
                                 static_cast<const void*>(sourceNode->_t),
                                 static_cast<const void*>(destNode->_t),
                                 static_cast<void*>(destNode->_t));
        }
    }

    if (srcB > 0) {
        if (sourceNode->_lastActivations && destNode->_lastActivations) {
            ObliviousBytesAssign(pred,
                                 bs_float_b,
                                 static_cast<const void*>(sourceNode->_lastActivations),
                                 static_cast<const void*>(destNode->_lastActivations),
                                 static_cast<void*>(destNode->_lastActivations));
        }
        if (sourceNode->_lastDeltaforBPs && destNode->_lastDeltaforBPs) {
            ObliviousBytesAssign(pred,
                                 bs_float_b,
                                 static_cast<const void*>(sourceNode->_lastDeltaforBPs),
                                 static_cast<const void*>(destNode->_lastDeltaforBPs),
                                 static_cast<void*>(destNode->_lastDeltaforBPs));
        }
        if (sourceNode->_lastGradients && destNode->_lastGradients) {
            ObliviousBytesAssign(pred,
                                 bs_float_b,
                                 static_cast<const void*>(sourceNode->_lastGradients),
                                 static_cast<const void*>(destNode->_lastGradients),
                                 static_cast<void*>(destNode->_lastGradients));
        }
        if (sourceNode->_ActiveinputIds && destNode->_ActiveinputIds) {
            ObliviousBytesAssign(pred,
                                 bs_int_b,
                                 static_cast<const void*>(sourceNode->_ActiveinputIds),
                                 static_cast<const void*>(destNode->_ActiveinputIds),
                                 static_cast<void*>(destNode->_ActiveinputIds));
        }
    }
}

float Node::getLastActivation(int inputID)
{
    int active = obl::OblEq(_ActiveinputIds[inputID], 1);
    float val  = _lastActivations[inputID];

    // 如果 active==1，则返回 val；否则返回 0.0f
    float res = obl::OblChoose(active, val, 0.0f);
    return res;
}

void Node::incrementDelta(int inputID, float incrementValue)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));
    float cur  = _lastDeltaforBPs[inputID];
    float cand = cur + incrementValue;

    // isPos = 1  当 lastActivation > 0， 否则 0
    int isPos = obl::OblGt(_lastActivations[inputID], 0.0f);

    // 如果 isPos==1，则选 cand；否则选 cur
    _lastDeltaforBPs[inputID] = obl::OblChoose(isPos, cand, cur);
}

float Node::getActivation(int* indices, float* values, int length, int inputID)
{

    if (inputID>_currentBatchsize){
        assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));
    }

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

     // 1) 归一化成 softmax 概率 p_u
    float p = _lastActivations[inputID] / (normalizationConstant + 1e-7f);
    _lastActivations[inputID] = p;
    _lastGradients[inputID]   = 1.0f;

    // 2) Oblivious 地判断 u 是否在 label 集合中
    int isPos = 0;
    for (int idx = 0; idx < labelsize; ++idx) {
        int eq = obl::OblEq(label[idx], _IDinLayer);
        // isPos = isPos || eq; 用 OblChoose 写成无分支：
        isPos = obl::OblChoose(eq, 1, isPos);
    }

    // 3) 计算正类/负类的 delta 值，再用 OblChoose 合并
    float posDelta = (1.0f / labelsize - p) / _currentBatchsize;
    float negDelta = (-p) / _currentBatchsize;

    _lastDeltaforBPs[inputID] = obl::OblChoose(isPos, posDelta, negDelta);
}

void Node::backPropageSingle(Node* previousNode, float learningRate, int inputID) {
    float delta = _lastDeltaforBPs[inputID];
    int prevID = previousNode->_IDinLayer;

    previousNode->incrementDelta(inputID, _lastDeltaforBPs[inputID]* _weights[previousNode->_IDinLayer]);
    float grad_t = _lastDeltaforBPs[inputID] * previousNode->getLastActivation(inputID);
    float grad_tsq = grad_t * grad_t;

    if (ADAM)
    {
        _t[previousNode->_IDinLayer] += grad_t;
    }
    else
    {
        _mirrorWeights[previousNode->_IDinLayer] += learningRate * grad_t;
    }

    // Bias accumulation is handled by backPropageBias(),
    // which the caller invokes once per (this-slot, inputID).

}

void Node::backPropageBias(float learningRate, int inputID) {
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

Node::Node(const Node& other)
  : NodeScalarBlock(static_cast<const NodeScalarBlock&>(other)),
    owns_weights(1),
    _currentBatchsize(other._currentBatchsize),
    _dim(other._dim)
{
    if (other._lastActivations) {
        _lastActivations = new float[_currentBatchsize] ();
        std::copy(other._lastActivations, other._lastActivations + _currentBatchsize, _lastActivations);
    } else {
        _lastActivations = nullptr;
    }

    if (other._lastDeltaforBPs) {
        _lastDeltaforBPs = new float[_currentBatchsize];

        std::copy(other._lastDeltaforBPs, other._lastDeltaforBPs + _currentBatchsize, _lastDeltaforBPs);
    } else {
        _lastDeltaforBPs = nullptr;
    }

    if (other._lastGradients) {
        _lastGradients = new float[_currentBatchsize] ();
        std::copy(other._lastGradients, other._lastGradients + _currentBatchsize, _lastGradients);
    } else {
        _lastGradients = nullptr;
    }

    if (other._ActiveinputIds) {
        _ActiveinputIds = new int[_currentBatchsize] ();

        std::copy(other._ActiveinputIds, other._ActiveinputIds + _currentBatchsize, _ActiveinputIds);
    } else {
        _ActiveinputIds = nullptr;
    }

    if (ADAM && other._adamAvgMom) {
        _adamAvgMom = new float[_dim] ();

        std::copy(other._adamAvgMom, other._adamAvgMom + _dim, _adamAvgMom);
    } else {
        _adamAvgMom = nullptr;
    }

    if (ADAM && other._adamAvgVel) {
        _adamAvgVel = new float[_dim] ();
        std::copy(other._adamAvgVel, other._adamAvgVel + _dim, _adamAvgVel);
    } else {
        _adamAvgVel = nullptr;
    }

    if (ADAM && other._t) {
        _t = new float[_dim] ();

        std::copy(other._t, other._t + _dim, _t);
    } else {
        _t = nullptr;
    }

    if (other._weights) {
        _weights = new float[_dim] ();
        std::copy(other._weights, other._weights + _dim, _weights);
    } else {
        _weights = nullptr;
    }

}

Node& Node::operator=(const Node& other)
{
    if (this == &other)
        return *this;

    delete[] _lastActivations;
    delete[] _lastDeltaforBPs;
    delete[] _lastGradients;
    delete[] _ActiveinputIds;

    if (ADAM)
    {
        delete[] _adamAvgMom;
        delete[] _adamAvgVel;
        delete[] _t;
    }

    if (_weights && owns_weights)
        delete[] _weights;

    isDummy = other.isDummy;
    _type = other._type;
    _currentBatchsize = other._currentBatchsize;
    _dim = other._dim;
    _layerNum = other._layerNum;
    _IDinLayer = other._IDinLayer;
    _bias = other._bias;
    _tbias = other._tbias;
    _adamAvgMombias = other._adamAvgMombias;
    _adamAvgVelbias = other._adamAvgVelbias;
    _mirrorbias = other._mirrorbias;
    _lshIndex = other._lshIndex;
    _isOverflow = other._isOverflow;

    if (other._lastActivations) {
        _lastActivations = new float[_currentBatchsize];
        std::copy(other._lastActivations, other._lastActivations + _currentBatchsize, _lastActivations);
    } else {
        _lastActivations = nullptr;
    }

    if (other._lastDeltaforBPs) {
        _lastDeltaforBPs = new float[_currentBatchsize];
        std::copy(other._lastDeltaforBPs, other._lastDeltaforBPs + _currentBatchsize, _lastDeltaforBPs);
    } else {
        _lastDeltaforBPs = nullptr;
    }

    if (other._lastGradients) {
        _lastGradients = new float[_currentBatchsize];
        std::copy(other._lastGradients, other._lastGradients + _currentBatchsize, _lastGradients);
    } else {
        _lastGradients = nullptr;
    }

    if (other._ActiveinputIds) {
        _ActiveinputIds = new int[_currentBatchsize];
        std::copy(other._ActiveinputIds, other._ActiveinputIds + _currentBatchsize, _ActiveinputIds);
    } else {
        _ActiveinputIds = nullptr;
    }

    if (ADAM && other._adamAvgMom) {
        _adamAvgMom = new float[_dim];
        std::copy(other._adamAvgMom, other._adamAvgMom + _dim, _adamAvgMom);
    } else {
        _adamAvgMom = nullptr;
    }

    if (ADAM && other._adamAvgVel) {
        _adamAvgVel = new float[_dim];
        std::copy(other._adamAvgVel, other._adamAvgVel + _dim, _adamAvgVel);
    } else {
        _adamAvgVel = nullptr;
    }

    if (ADAM && other._t) {
        _t = new float[_dim];
        std::copy(other._t, other._t + _dim, _t);
    } else {
        _t = nullptr;
    }

    if (other._weights) {
        _weights = new float[_dim];
        std::copy(other._weights, other._weights + _dim, _weights);
    } else {
        _weights = nullptr;
    }

    return *this;
}

Node::Node(Node&& other) noexcept
  : NodeScalarBlock(static_cast<const NodeScalarBlock&>(other)),  // 标量无所谓 move/copy
    owns_weights(other.owns_weights),
    _currentBatchsize(other._currentBatchsize),
    _dim(other._dim),
    _lastGradients(other._lastGradients),
    _ActiveinputIds(other._ActiveinputIds),
    _lastDeltaforBPs(other._lastDeltaforBPs),
    _lastActivations(other._lastActivations),
    _weights(other._weights),
    _mirrorWeights(other._mirrorWeights),
    _adamAvgMom(other._adamAvgMom),
    _adamAvgVel(other._adamAvgVel),
    _t(other._t),
    _update(other._update)
{
    other._lastGradients = nullptr;
    other._ActiveinputIds = nullptr;
    other._lastDeltaforBPs = nullptr;
    other._lastActivations = nullptr;
    other._weights = nullptr;
    other._mirrorWeights = nullptr;
    other._adamAvgMom = nullptr;
    other._adamAvgVel = nullptr;
    other._t = nullptr;
    other._update = nullptr;

    other.owns_weights   = 0;       // 让被搬空的对象变成“空壳（不拥有资源）”
    other._dim           = 0;
    other._currentBatchsize = 0;
}

Node& Node::operator=(Node&& other) noexcept
{
    if (this == &other)
        return *this;

    delete[] _lastActivations;
    delete[] _lastDeltaforBPs;
    delete[] _lastGradients;
    delete[] _ActiveinputIds;

    if (ADAM)
    {
        delete[] _adamAvgMom;
        delete[] _adamAvgVel;
        delete[] _t;
    }

    if (_weights && owns_weights)
        delete[] _weights;

    owns_weights      = other.owns_weights;
    isDummy = other.isDummy;
    _type = other._type;
    _currentBatchsize = other._currentBatchsize;
    _dim = other._dim;
    _layerNum = other._layerNum;
    _IDinLayer = other._IDinLayer;
    _bias = other._bias;
    _tbias = other._tbias;
    _adamAvgMombias = other._adamAvgMombias;
    _adamAvgVelbias = other._adamAvgVelbias;
    _mirrorbias = other._mirrorbias;

    _lastGradients = other._lastGradients;
    _ActiveinputIds = other._ActiveinputIds;
    _lastDeltaforBPs = other._lastDeltaforBPs;
    _lastActivations = other._lastActivations;

    _weights = other._weights;
    _mirrorWeights = other._mirrorWeights;
    _adamAvgMom = other._adamAvgMom;
    _adamAvgVel = other._adamAvgVel;
    _t = other._t;
    _update = other._update;

    _lshIndex = other._lshIndex;
    _isOverflow = other._isOverflow;

    other._lastGradients = nullptr;
    other._ActiveinputIds = nullptr;
    other._lastDeltaforBPs = nullptr;
    other._lastActivations = nullptr;

    other._weights = nullptr;
    other._mirrorWeights = nullptr;
    other._adamAvgMom = nullptr;
    other._adamAvgVel = nullptr;
    other._t = nullptr;
    other._update = nullptr;

    return *this;
}

Node::~Node()
{

    if (_lastActivations)
        delete[] _lastActivations;

    if (_lastDeltaforBPs)
        delete[] _lastDeltaforBPs;

    if (_lastGradients)
        delete[] _lastGradients;

    if (_ActiveinputIds)
        delete[] _ActiveinputIds;

    if (ADAM)
    {
        if(owns_weights){
            if (_adamAvgMom)
                delete[] _adamAvgMom;
            if (_adamAvgVel)
                delete[] _adamAvgVel;
        }

        if (_t)
            delete[] _t;
    }

    if (owns_weights && _weights)
        delete[] _weights;
}

