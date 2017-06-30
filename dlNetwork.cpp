#include "dlNetwork.h"
#include "dlFullyConnectedLayer.h"
#include "debug.h"
#include <fstream>
#include <iostream>
dlNetwork::dlNetwork(char * name) :
	m_name(name)
{
}

dlNetwork::~dlNetwork()
{
	for (size_t i = 0; i < m_layers.size(); i++)
	{
		delete m_layers[i];
		m_layers[i] = NULL;
	}
}
dlFullyConnectedLayer * dlNetwork::GetLayer(uint id)
{
	DBG_ASSERT(id < m_layers.size(), "Layer id is greater than layer numbers!");
	return m_layers[id];
}

double sigmoid(double x)
{ 
	return (1 / (1 + exp(-x))); 
}

void dlNetwork::Init()
{
	dlFullyConnectedLayer* inputLayer = new dlFullyConnectedLayer(0, 28 * 28, NULL, NULL, NULL);
	m_layers.push_back(inputLayer);
	dlFullyConnectedLayer* hiddenLayer = new dlFullyConnectedLayer(28 * 28, 300, sigmoid, inputLayer, NULL);
	m_layers.push_back(hiddenLayer);
	dlFullyConnectedLayer* outputLayer = new dlFullyConnectedLayer(300, 10, sigmoid, hiddenLayer, NULL);
	m_layers.push_back(outputLayer);
	memset(&m_trainInfo, 0, sizeof(m_trainInfo));
	memset(&m_epoch, 0, sizeof(m_epoch));
	m_trainInfo.layerNum = 3;
}

void dlNetwork::LoadInfo()
{
	string path = m_name + ".dat";
	ifstream file(path.c_str(), ios::binary);

	uint inputSize, outputSize;
	dlFullyConnectedLayer* newLayer;
	dlFullyConnectedLayer* lastLayer = NULL;
	file.read((char*)&m_trainInfo, sizeof(m_trainInfo));
	for (size_t i = 0; i < m_trainInfo.layerNum; i++)
	{		
		file.read((char*)&inputSize, sizeof(uint));
		file.read((char*)&outputSize, sizeof(uint));
		newLayer = new dlFullyConnectedLayer(inputSize, outputSize, sigmoid, lastLayer, NULL);
		if (i > 0)
		{
			file.read((char*)newLayer->m_weight.data(), inputSize * outputSize * sizeof(float));
			file.read((char*)newLayer->m_blas.data(), outputSize * sizeof(float));
		}
		m_layers.push_back(newLayer);
		lastLayer = newLayer;
	}
	memset(&m_epoch, 0, sizeof(m_epoch));
}

void dlNetwork::SaveInfo()
{
	string path = m_name + ".dat";
	ofstream file(path.c_str(), ios::binary);
	file.write((char*)&m_trainInfo, sizeof(m_trainInfo));
	for (size_t i = 0; i < m_trainInfo.layerNum; i++)
	{
		file.write((char*)&m_layers[i]->m_inputSize, sizeof(uint));
		file.write((char*)&m_layers[i]->m_outputSize, sizeof(uint));
		if (i > 0)
		{
			file.write((char*)m_layers[i]->m_weight.data(), m_layers[i]->m_weight.size() * sizeof(float));
			file.write((char*)m_layers[i]->m_blas.data(), m_layers[i]->m_blas.size() * sizeof(float));
		}
	}
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

void dlNetwork::CalGradient(MatrixXf & input, MatrixXf & target)
{
	m_layers[0]->SetInputData(input);
	for (size_t i = 1; i < m_trainInfo.layerNum; i++)
	{
		m_layers[i]->CalOutput();
	}
	MatrixXf data = target;
	MatrixXf Gradients;
	MatrixXf delta = MatrixXf::Zero(m_layers[2]->m_outputSize, 1);
	m_layers[2]->CalDelta(data, delta);
	Gradients = delta * m_layers[1]->m_output.transpose();
	double epsilon = 0.0001;
	double ed = 0;
	for (size_t i = 0; i < m_layers[2]->m_weight.rows(); i++)
	{
		for (size_t j = 0; j < m_layers[2]->m_weight.cols(); j++)
		{
			m_layers[2]->m_weight(i, j) = m_layers[2]->m_weight(i, j) + epsilon;
			m_layers[1]->CalOutput();
			m_layers[2]->CalOutput();
			for (size_t o = 0; o < m_layers[2]->m_outputSize; o++)
			{
				ed += (data(o, 0) - m_layers[2]->m_output(o, 0)) * (data(o, 0) - m_layers[2]->m_output(o, 0));
			}
			m_layers[2]->m_weight(i, j) = m_layers[2]->m_weight(i, j) - 2 *  epsilon;
			m_layers[1]->CalOutput();
			m_layers[2]->CalOutput();
			for (size_t o = 0; o < m_layers[2]->m_outputSize; o++)
			{
				ed -= (data(o, 0) - m_layers[2]->m_output(o, 0)) * (data(o, 0) - m_layers[2]->m_output(o, 0));
			}
			m_layers[2]->m_weight(i, j) = m_layers[2]->m_weight(i, j) + epsilon;			
			cout << ed / 4 / epsilon << " " << Gradients(i,j) <<endl;
			ed = 0;
		}
	}
}

void dlNetwork::Train(MatrixXf& input, MatrixXf& target)
{
	m_layers[0]->SetInputData(input);
	for (size_t i = 1; i < m_trainInfo.layerNum; i++)
	{
		m_layers[i]->CalOutput();
	}
	MatrixXf data = target;
	for (size_t i = m_trainInfo.layerNum - 1; i >= 1; i--)
	{
		MatrixXf delta = MatrixXf::Zero(m_layers[i]->m_outputSize, 1);
		m_layers[i]->CalDelta(data, delta);
		m_layers[i]->UpdateWB(delta);
		data = delta;
	}
	MatrixXf::Index maxRow, maxCol;
	target.maxCoeff(&maxRow, &maxCol);
	uint targetValue = maxRow;
	m_layers[m_trainInfo.layerNum - 1]->m_output.maxCoeff(&maxRow, &maxCol);
	if (maxRow != targetValue)
	{
		m_trainInfo.faults++;
		m_epoch.faults++;
	}
	m_trainInfo.trainTimes++;
	m_epoch.times++;
}

void dlNetwork::Test(MatrixXf & input, MatrixXf & target)
{
	m_layers[0]->SetInputData(input);
	for (size_t i = 1; i < m_trainInfo.layerNum; i++)
	{
		m_layers[i]->CalOutput();
	}
	MatrixXf::Index maxRow, maxCol;
	target.maxCoeff(&maxRow, &maxCol);
	uint targetValue = maxRow;
	m_layers[m_trainInfo.layerNum - 1]->m_output.maxCoeff(&maxRow, &maxCol);
	if (maxRow != targetValue)
	{
		m_epoch.faults++;
	}
	m_epoch.times++;
}
