#pragma once
#include "dlLayer.h"

class dlConvLayer : public dlLayer
{
public:
	dlConvLayer(dlLayerType type, Vector3i inDim, Vector3i filterDim, uint filterNum, 
		uint zeroPadding, FunActivator funActivator, dlLayer* pUpLayer);
private:
	Vector3i m_filterDim;
};