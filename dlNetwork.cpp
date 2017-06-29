#include "dlNetwork.h"
#include "dlFullyConnectedLayer.h"
#include "debug.h"
#include <fstream>
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
		newLayer = new dlFullyConnectedLayer(0, inputSize * outputSize, NULL, lastLayer, NULL);		
		if (i > 0)
		{
			file.read((char*)newLayer->m_weight.data(), inputSize * outputSize * sizeof(float));
			file.read((char*)newLayer->m_blas.data(), inputSize * outputSize * sizeof(float));
		}
		m_layers.push_back(newLayer);
		lastLayer = newLayer;
	}
}

void dlNetwork::SaveInfo()
{
	string path = m_name + ".dat";
	ofstream file(path.c_str(), ios::binary);
	file.write((char*)&m_trainInfo, sizeof(m_trainInfo));
	for (size_t i = 0; i < m_trainInfo.layerNum; i++)
	{
		file.write((char*)&m_layers[0]->m_inputSize, sizeof(uint));
		file.write((char*)&m_layers[0]->m_outputSize, sizeof(uint));
		if (i > 0)
		{
			file.write((char*)m_layers[0]->m_weight.data(), m_layers[0]->m_weight.size() * sizeof(float));
			file.write((char*)m_layers[0]->m_blas.data(), m_layers[0]->m_blas.size() * sizeof(float));
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
	m_trainInfo.trainTimes++;
	//m_trainInfo.corrects
}
