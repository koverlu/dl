#include "dlConvLayer.h"
#include "debug.h"
#include "Conv.h"

dlConvLayer::dlConvLayer(dlLayerType type, Vector3i inDim, Vector3i filterDim, uint filterNum,
	uint zeroPadding, ActivatorType activator, dlLayer * pUpLayer) :
	dlLayer(type, inDim, Vector3i(0, 0, 0), activator, pUpLayer),
	m_filterDim(filterDim)
{
	DBG_ASSERT(inDim.x() >= filterDim.x() && inDim.y() >= filterDim.y() && inDim.z() == filterDim.z(),
		"Wrong filter dimension !\n");
	m_outDim.x() = inDim.x() + 2 * zeroPadding - filterDim.x() + 1;
	m_outDim.y() = inDim.y() + 2 * zeroPadding - filterDim.y() + 1;
	m_outDim.z() = filterNum;

	for (uint i = 0; i < filterNum; i++)
	{
		dlFilter newfilter;
		m_vFilter.push_back(newfilter);
		for (uint j = 0; j < m_filterDim.z(); j++)
			m_vFilter[i].weights.push_back(MatrixXf::Random(m_filterDim.x(), m_filterDim.y()) * 0.1f);

		m_vData.push_back(MatrixXf::Zero(m_outDim.x(), m_outDim.y()));
	}
}

void dlConvLayer::Forward()
{
	for (size_t i = 0; i < m_outDim.z(); i++)
	{
		for (size_t j = 0; j < m_filterDim.z(); j++)
		{
			MatrixXf convMat = Conv(m_pUpLayer->m_vData[j], m_vFilter[i].weights[j]);
			MatrixXf oneMat = m_vData[i];
			m_vData[i] += EleWiseOp(convMat, m_Activator) + m_vFilter[i].bias * oneMat;
		}
	}
}

void dlConvLayer::Backward()
{
	for (size_t i = 0; i < m_outDim.z(); i++)
	{
		m_vDeviation[i].setZero();
		for (size_t j = 0; j < m_pDownLayer->m_outDim.z(); j++)
		{
			MatrixXf tmp_mat = Conv(Padding(m_pDownLayer->m_vDeviation[j], 1, 1), FlipV(m_pDownLayer->m_vFilter[j].weights[i]));
			m_vDeviation[i] += tmp_mat.cwiseProduct(m_vData[i]);
		}
		for (size_t k = 0; k < m_filterDim.z(); k++)
		{
			MatrixXf filter_deriv = Conv(m_pUpLayer->m_vData[k], m_vDeviation[i]);
			m_vFilter[i].weights[k] -= m_rate * filter_deriv;
		}
		m_vFilter[i].bias -= m_rate * m_vDeviation[i].sum();
	}
}
