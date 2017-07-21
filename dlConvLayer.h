#pragma once
#include "dlLayer.h"

class dlConvLayer : public dlLayer
{
public:
	dlConvLayer(dlLayerType type, Vector3i inDim, Vector3i filterDim, uint filterNum, 
		uint zeroPadding, ActivatorType activator, dlLayer* pUpLayer);
	virtual void Forward();
	virtual void Backward();
};