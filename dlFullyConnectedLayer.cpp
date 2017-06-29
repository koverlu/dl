#include "dlFullyConnectedLayer.h"
#include "debug.h"
dlFullyConnectedLayer::dlFullyConnectedLayer(uint inputSize, uint outputSize, FunActivator funActivator, 
	dlFullyConnectedLayer* pUpLayer, dlFullyConnectedLayer* pDownLayer) :
	m_inputSize(inputSize),
	m_outputSize(outputSize),
	m_Activator(funActivator),
	m_pUpLayer(pUpLayer),
	m_pDownLayer(pDownLayer)	
{
	if (pUpLayer)
	{
		if (pUpLayer->m_pDownLayer == NULL)
			pUpLayer->m_pDownLayer = this;
		else
			DBG_ASSERT(pUpLayer->m_pDownLayer == this, "UpLayer set error!");
	}
	if (pDownLayer)
	{
		if (pDownLayer->m_pUpLayer == NULL)
			pDownLayer->m_pUpLayer = this;
		else
			DBG_ASSERT(pDownLayer->m_pUpLayer == this, "DownLayer set error!");
	}

	m_blas = MatrixXf::Zero(m_outputSize, 1);
	m_output = MatrixXf::Zero(m_outputSize, 1);
	if (inputSize > 0)
	{
		//m_weight(OUTPUT, INPUT)
		m_weight = MatrixXf::Zero(m_outputSize, inputSize);
	}
	m_rate = 0.1f;
}

dlFullyConnectedLayer::~dlFullyConnectedLayer()
{
}

void dlFullyConnectedLayer::CalOutput()
{
	DBG_ASSERT(m_pUpLayer, "It's an input layer!");
	m_output = m_weight * m_pUpLayer->m_output + m_blas;
	for (size_t i = 0; i < m_output.rows(); i++)
	{
		m_output(i, 0) = m_Activator(m_output(i, 0));
	}
}

void dlFullyConnectedLayer::SetInputData(MatrixXf& data)
{
	m_output = data;
}

void dlFullyConnectedLayer::UpdateWB(MatrixXf& delta)
{
	m_weight = m_weight + m_rate *	delta * m_pUpLayer->m_output.transpose();
	m_blas = m_blas + m_rate * delta;
}

void dlFullyConnectedLayer::CalDelta(MatrixXf& data, MatrixXf& dst_delta)
{
	if (m_pDownLayer)
	{
		// "data" here means the delta of DownLayer
		for (uint i = 0; i < m_outputSize; i++)
		{			
			double sum = 0;
			for (uint j = 0; j < m_pDownLayer->m_outputSize; j++)
			{
				sum += m_pDownLayer->m_weight(j, i) * data(j, 0);
			}
			dst_delta(i, 0) = m_output(i, 0) * (1 - m_output(i, 0)) * sum;
		}
	}
	else
	{
		// For output layer, "data" is the target value
		for (uint i = 0; i < m_output.rows(); i++)
		{
			dst_delta(i, 0) = m_output(i, 0) * (1 - m_output(i, 0)) * (data(i, 0) - m_output(i, 0));
		}
	}

}

void dlFullyConnectedLayer::SetWB(MatrixXf& weight, MatrixXf& blas)
{
	m_weight = weight;
	m_blas = blas;
}
