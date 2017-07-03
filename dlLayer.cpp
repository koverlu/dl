#include "dlLayer.h"
#include "debug.h"

dlLayer::dlLayer(dlLayerType type, Vector3i inDim, Vector3i outDim, FunActivator funActivator, dlLayer* pUpLayer) :
	m_type(type),
	m_inDim(inDim),
	m_outDim(outDim),
	m_Activator(funActivator),
	m_pUpLayer(pUpLayer)
{
	if (pUpLayer)
	{
		if (pUpLayer->m_pDownLayer == NULL)
			pUpLayer->m_pDownLayer = this;
		else
			DBG_ASSERT(pUpLayer->m_pDownLayer == this, "UpLayer set error!");
	}
}
