#pragma once
#include <vector>
#include "dlDef.h"

class dlNode;
class dlNetwork;
class dlConnection;

typedef double(*FunActivator)(double);

enum dlLayerType
{
	DL_INPUT,
	DL_HIDDEN,
	DL_OUTPUT
};

enum ActivatorType
{
	ACTIV_SIGMOID,
	ACTIV_RELU,
	ACTIV_NONE
};

template<typename T>
T EleWiseOp(const T& matrixA, FunActivator op)
{
	T matrixR;
	size_t r_row = matrixA.rows();
	size_t r_col = matrixA.cols();
	matrixR.resize(r_row, r_col);
	for (size_t i = 0; i < r_row; i++)
	{
		for (size_t j = 0; j < r_col; j++)
		{
			matrixR(i, j) = op(matrixA(i, j));
		}
	}
	return matrixR;
}

struct dlFilter
{
	vector<MatrixXf> weights;
	double bias;
};

class dlLayer
{
public:
	dlLayer(dlLayerType type, Vector3i inDim, Vector3i filterDim, ActivatorType activator, dlLayer* pUpLayer);
	virtual void Forward() = 0;
	virtual void Backward() = 0;
	void Save();
	void GradientCheck();
	void SetFilter(uint idx, dlFilter& flt);
	dlLayer* m_pUpLayer;
	dlLayer* m_pDownLayer;
	uint m_layerId;
	dlLayerType m_type;
	Vector3i m_inDim;
	Vector3i m_outDim;
	Vector3i m_filterDim;
	MatrixXf m_bias;
	vector<dlFilter> m_vFilter;
	vector<MatrixXf> m_vData;
	vector<MatrixXf> m_vDeviation;
	FunActivator m_Activator;
	double m_rate;
private:
	static uint m_sLayerCnt;
};

class dlInputLayer : public dlLayer
{
public:
	dlInputLayer(Vector3i outDim);
	void SetInputData(uint idx, const MatrixXf& mat);
	virtual void Forward() {}
	virtual void Backward() {}
};