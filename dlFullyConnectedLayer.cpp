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

	m_blas = new double[m_outputSize];
	m_output = new double[m_outputSize];
	if (inputSize > 0)
	{
		m_weight = new double*[m_inputSize];
		for (uint i = 0; i < m_inputSize; i++)
		{
			m_weight[i] = new double[m_outputSize];
		}
	}
	else
		m_weight = NULL;
}

dlFullyConnectedLayer::~dlFullyConnectedLayer()
{
	if(m_blas)
		delete[] m_blas;
	if (m_output)
		delete[] m_output;
	if (m_weight)
	{
		for (size_t i = 0; i < m_inputSize; i++)
			delete[] m_weight[i];
		delete[] m_weight;
	}
}

void dlFullyConnectedLayer::CalOutput()
{
	DBG_ASSERT(m_pUpLayer, "It's an input layer!");
	for (uint i = 0; i < m_outputSize; i++)
	{
		m_output[i] = 0;
		for (uint j = 0; j < m_inputSize; j++)
			m_output[i] += m_pUpLayer->m_output[j] * m_weight[j][i];
		m_output[i] += m_blas[i];
		m_output[i] = m_Activator(m_output[i]);
	}
}

void dlFullyConnectedLayer::SetInputData(double * data)
{
	memcpy(m_output, data, m_inputSize * sizeof(double));
}

void dlFullyConnectedLayer::UpdateWB(double * delta)
{
	for (uint i = 0; i < m_outputSize; i++)
	{
		for (uint j = 0; j < m_inputSize; j++)
		{
			m_weight[i][j] = m_weight[i][j] + m_rate * delta[i] * m_pUpLayer->m_output[j];
		}
		m_blas[i] = m_blas[i] + m_rate * delta[i];
	}
}

void dlFullyConnectedLayer::CalSigma(double* data, double* dst_delta)
{
	if (m_pDownLayer)
	{
		for (uint i = 0; i < m_outputSize; i++)
		{
			dst_delta[i] = m_output[i] * (1 - m_output[i]);
			for (uint j = 0; j < m_pDownLayer->m_outputSize; j++)
			{
				dst_delta[i] += m_pDownLayer->m_weight[i][j] * data[i];
			}
		}
	}
	else
	{
		// For output layer
		for (uint i = 0; i < m_outputSize; i++)
		{
			dst_delta[i] = m_output[i] * (1 - m_output[i]) * (data[i] - m_output[i]);
		}
	}

}