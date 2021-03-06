#include "network.h"
#include "fc.h"
#include "lstm.h"
#include "debug.h"
#include "common.h"
#include <fstream>
#include <iostream>
dlNetwork::dlNetwork(char * name, uint batchSize) :
	m_name(name),
	m_batchSize(batchSize)
{
	m_thousandsFaults = 0;
}

dlNetwork::~dlNetwork()
{

}

void dlNetwork::Init()
{
	m_pLSTMLayer = new LSTMLayer(28, 128, m_batchSize, 28, 0.001);
	m_pFCLayer = new FCLayer(128, 10, m_batchSize, 0.001);
	m_pFCLayer->m_pInputs = &m_pLSTMLayer->m_output;
	m_pLSTMLayer->m_pBackDeltas = &m_pFCLayer->m_back_deltas;
	memset(&m_trainInfo, 0, sizeof(m_trainInfo));
	memset(&m_epoch, 0, sizeof(m_epoch));
	m_trainInfo.layerNum = 2;
}

void dlNetwork::LoadInfo()
{
	string path = m_name + ".dat";
	ifstream file(path.c_str(), ios::binary);

	uint inputSize, outputSize;
	file.read((char*)&m_trainInfo, sizeof(m_trainInfo));
	for (size_t i = 0; i < m_trainInfo.layerNum; i++)
	{		
		file.read((char*)&inputSize, sizeof(uint));
		file.read((char*)&outputSize, sizeof(uint));
		if (i > 0)
		{
			//file.read((char*)newLayer->m_weight.data(), inputSize * outputSize * sizeof(float));
			//file.read((char*)newLayer->m_blas.data(), outputSize * sizeof(float));
		}
	}
	memset(&m_epoch, 0, sizeof(m_epoch));
}

void dlNetwork::SaveInfo()
{
	string path = m_name + ".dat";
	ofstream file(path.c_str(), ios::binary);
	file.write((char*)&m_trainInfo, sizeof(m_trainInfo));
	//Write LSTM Layer
	file.write((char*)&m_pLSTMLayer->m_inVecLen, sizeof(uint));
	file.write((char*)&m_pLSTMLayer->m_stateLen, sizeof(uint));
	file.write((char*)&m_pLSTMLayer->m_steps, sizeof(uint));
	file.write((char*)&m_pLSTMLayer->m_weights, m_pLSTMLayer->m_weights.size() * sizeof(double));
	file.write((char*)&m_pLSTMLayer->m_bias, m_pLSTMLayer->m_bias.size() * sizeof(double));
	//Write FC Layer
	file.write((char*)&m_pFCLayer->m_inVecLen, sizeof(uint));
	file.write((char*)&m_pFCLayer->m_stateLen, sizeof(uint));
	file.write((char*)&m_pFCLayer->m_weights, m_pFCLayer->m_weights.size() * sizeof(double));
	file.write((char*)&m_pFCLayer->m_bias, m_pFCLayer->m_bias.size() * sizeof(double));
}

double dlNetwork::EpochStatistics()
{
	double errorRate = (double)m_epoch.faults / m_epoch.times;
	DBG_PRINT("EPOCH %d total ER: %f\n", m_epochVector.size(), errorRate);
	m_epochVector.push_back(m_epoch);
	memset(&m_epoch, 0, sizeof(m_epoch));
	return errorRate;
}

double dlNetwork::GetLastErrorRate(uint num)
{
	uint size = m_epochVector.size();
	uint errorSum = 0;
	uint timsSum = 0;
	if (size < num)
	{
		for (size_t i = 0; i < size; i++)
		{
			errorSum += m_epochVector[i].faults;
			timsSum += m_epochVector[i].times;
		}
	}
	else
	{
		for (size_t i = size - 1; i >= size - num; i--)
		{
			errorSum += m_epochVector[i].faults;
			timsSum += m_epochVector[i].times;
		}
	}
	double errorRate = (double)errorSum / timsSum;
	DBG_PRINT("Last %d epoch ER %f\n", size < num ? size : num, errorRate);
	return  errorRate;
}

void dlNetwork::Train()
{
	m_pLSTMLayer->Forward();
	m_pFCLayer->Forward();
	m_pFCLayer->BackWard();
	m_pLSTMLayer->BackWard();
	for (uint b = 0; b < m_batchSize; b++)
	{
		uint targetValue = VectorMaxIdx(*m_pFCLayer->m_pTargets, 10, b * 10);
		uint OutputValue = VectorMaxIdx(m_pFCLayer->m_states, 10, b * 10);
		if (OutputValue != targetValue)
		{
			m_trainInfo.faults++;
			m_epoch.faults++;
			m_thousandsFaults++;
		}
		m_epoch.times++;
	}

	m_trainInfo.trainTimes++;
	uint period = m_batchSize == 1 ? 1000 : m_batchSize * 10;
	if (m_epoch.times % period == 0)
	{
		DBG_PRINT("Epoch %d: %d / %d, Mini-Batch ER: %f, Total ER: %f, W0 = %f\n", 
			m_epochVector.size(), m_epoch.faults, m_epoch.times, (double)m_thousandsFaults / period, (double)m_epoch.faults / m_epoch.times, m_pLSTMLayer->m_weights[0]);
		m_thousandsFaults = 0;
	}
		
		
}

void dlNetwork::Test()
{
	m_pLSTMLayer->Forward();
	m_pFCLayer->Forward();
	uint targetValue = VectorMaxIdx(*m_pFCLayer->m_pTargets);
	uint OutputValue = VectorMaxIdx(m_pFCLayer->m_states);
	if (OutputValue != targetValue)
	{
		m_epoch.faults++;
	}
	m_epoch.times++;
}

void dlNetwork::SetInputsAndTargets(vector<double>* pInputs, vector<double>* pTargets)
{
	m_pLSTMLayer->m_pInputs = pInputs;
	m_pFCLayer->m_pTargets = pTargets;
}
