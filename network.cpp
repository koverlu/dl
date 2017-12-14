#include "network.h"
#include "fc.h"
#include "lstm.h"
#include "debug.h"
#include "common.h"
#include <fstream>
#include <iostream>
dlNetwork::dlNetwork(char * name) :
	m_name(name)
{
	m_thousandsFaults = 0;
}

dlNetwork::~dlNetwork()
{

}

void dlNetwork::Init()
{
	m_pLSTMLayer = new LSTMLayer(28, 128, 1, 28, 0.01);
	m_pFCLayer = new FCLayer(128, 10, 1, 0.01);
	m_pFCLayer->m_pInputs = &m_pLSTMLayer->m_output;
	m_pLSTMLayer->m_pBackDeltas = &m_pFCLayer->m_wei_grad;
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
	DBG_PRINT("%f\n", errorRate);
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
	DBG_PRINT("@@@ %f\n", errorRate);
	return  errorRate;
}

//void dlNetwork::CalGradient(MatrixXf & input, MatrixXf & target)
//{
//	m_layers[0]->SetInputData(input);
//	for (size_t i = 1; i < m_trainInfo.layerNum; i++)
//	{
//		m_layers[i]->CalOutput();
//	}
//	MatrixXf data = target;
//	MatrixXf Gradients;
//	MatrixXf delta = MatrixXf::Zero(m_layers[2]->m_outputSize, 1);
//	m_layers[2]->CalDelta(data, delta);
//	Gradients = delta * m_layers[1]->m_output.transpose();
//	double epsilon = 0.0001;
//	double ed = 0;
//	for (size_t i = 0; i < m_layers[2]->m_weight.rows(); i++)
//	{
//		for (size_t j = 0; j < m_layers[2]->m_weight.cols(); j++)
//		{
//			m_layers[2]->m_weight(i, j) = m_layers[2]->m_weight(i, j) + epsilon;
//			m_layers[1]->CalOutput();
//			m_layers[2]->CalOutput();
//			for (size_t o = 0; o < m_layers[2]->m_outputSize; o++)
//			{
//				ed += (data(o, 0) - m_layers[2]->m_output(o, 0)) * (data(o, 0) - m_layers[2]->m_output(o, 0));
//			}
//			m_layers[2]->m_weight(i, j) = m_layers[2]->m_weight(i, j) - 2 *  epsilon;
//			m_layers[1]->CalOutput();
//			m_layers[2]->CalOutput();
//			for (size_t o = 0; o < m_layers[2]->m_outputSize; o++)
//			{
//				ed -= (data(o, 0) - m_layers[2]->m_output(o, 0)) * (data(o, 0) - m_layers[2]->m_output(o, 0));
//			}
//			m_layers[2]->m_weight(i, j) = m_layers[2]->m_weight(i, j) + epsilon;			
//			cout << ed / 4 / epsilon << " " << Gradients(i,j) <<endl;
//			ed = 0;
//		}
//	}
//}

void dlNetwork::Train()
{
	m_pLSTMLayer->Forward();
	m_pFCLayer->Forward();
	m_pFCLayer->BackWard();
	m_pLSTMLayer->BackWard();
	uint targetValue = VectorMaxIdx(*m_pFCLayer->m_pTargets);
	uint OutputValue = VectorMaxIdx(m_pFCLayer->m_states);
	if (OutputValue != targetValue)
	{
		m_trainInfo.faults++;
		m_epoch.faults++;
		m_thousandsFaults++;
	}
	m_trainInfo.trainTimes++;
	m_epoch.times++;
	if (m_epoch.times % 1000 == 0)
	{
		DBG_PRINT("Epoch %d: %d / %d, Thousands faults: %d, Total ER: %f\n", 
			m_epochVector.size(), m_epoch.faults, m_epoch.times, m_thousandsFaults, (double)m_epoch.faults / m_epoch.times);
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
