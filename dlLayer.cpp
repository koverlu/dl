#include "dlLayer.h"
#include "debug.h"
double activ_sigmoid(double x)
{
	return (1 / (1 + exp(-x)));
}

double activ_relu(double x)
{
	return x >= 0 ? x : 0;
}

uint dlLayer::m_sLayerCnt = 0;

dlLayer::dlLayer(dlLayerType type, Vector3i inDim, Vector3i outDim, ActivatorType activator, dlLayer* pUpLayer) :
	m_type(type),
	m_inDim(inDim),
	m_outDim(outDim),
	m_pUpLayer(pUpLayer),
	m_rate(0.1),
	m_layerId(m_sLayerCnt++)
{
	if (pUpLayer)
	{
		if (pUpLayer->m_pDownLayer == NULL)
			pUpLayer->m_pDownLayer = this;
		else
			DBG_ASSERT(pUpLayer->m_pDownLayer == this, "UpLayer set error!");
	}
	if (activator == ACTIV_SIGMOID)
	{
		m_Activator = activ_sigmoid;
	}
	else if (activator == ACTIV_RELU)
	{
		m_Activator = activ_relu;
	}
}
