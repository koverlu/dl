#include "dlNetwork.h"
#include "dlFullyConnectedLayer.h"
#include "debug.h"
dlLayer * dlNetwork::GetLayer(uint id)
{
	//DBG_ASSERT(id < m_layers.size(), "Layer id is greater than layer numbers!");
	//return m_layers[id];
}

double sigmoid(double x)
{ 
	return (1 / (1 + exp(-x))); 
}

void dlNetwork::Init()
{
	dlFullyConnectedLayer* inputLayer = new dlFullyConnectedLayer(0, 28 * 28, NULL, NULL, NULL);
	dlFullyConnectedLayer* hiddenLayer = new dlFullyConnectedLayer(28 * 28, 300, sigmoid, inputLayer, NULL);
	dlFullyConnectedLayer* outputLayer = new dlFullyConnectedLayer(300, 10, sigmoid, hiddenLayer, NULL);
	

}
