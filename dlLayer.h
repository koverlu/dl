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

class dlLayer
{
public:
	dlLayer(dlLayerType type, Vector3i inDim, Vector3i outDim, FunActivator funActivator, dlLayer* pUpLayer);

protected:
	dlLayer* m_pUpLayer;
	dlLayer* m_pDownLayer;
	uint m_layerId;
	dlLayerType m_type;
	Vector3i m_inDim;
	Vector3i m_outDim;
	MatrixXf m_blas;
	vector<MatrixXf> m_vWeight;
	vector<MatrixXf> m_output;
	FunActivator m_Activator;
};