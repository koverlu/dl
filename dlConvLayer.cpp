#include "dlConvLayer.h"
#include "debug.h"

dlConvLayer::dlConvLayer(dlLayerType type, Vector3i inDim, Vector3i filterDim, uint filterNum,
	uint zeroPadding, FunActivator funActivator, dlLayer * pUpLayer) :
	dlLayer(type, inDim, Vector3i(0, 0, 0), funActivator, pUpLayer),
	m_filterDim(filterDim)
{
	DBG_ASSERT(inDim.x() >= filterDim.x() && inDim.y() >= filterDim.y() && inDim.z() == filterDim.z(),
		"Wrong filter dimension !\n");
	m_outDim.x() = inDim.x() + 2 * zeroPadding - filterDim.x() + 1;
	m_outDim.y() = inDim.y() + 2 * zeroPadding - filterDim.y() + 1;
	m_outDim.z() = filterNum;

	for (uint i = 0; i < filterNum; i++)
	{
		for (uint j = 0; j < m_filterDim.z(); j++)
			m_vWeight.push_back(MatrixXf::Random(m_filterDim.x(), m_filterDim.y()) * 0.1f);

		m_output.push_back(MatrixXf::Zero(m_outDim.x(), m_outDim.y()));
	}
}
