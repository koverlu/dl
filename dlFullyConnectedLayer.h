#pragma once
#include "dlLayer.h"

typedef double(*FunActivator)(double);

class dlFullyConnectedLayer
{
public:
	dlFullyConnectedLayer(uint inputSize, uint outputSize, FunActivator funActivator, 
		dlFullyConnectedLayer* pUpLayer, dlFullyConnectedLayer* pDownLayer);
	~dlFullyConnectedLayer();	
	void CalOutput();
	void SetInputData(double* data);
	void UpdateWB(double* delta);
	void CalDelta(double* data, double* dst_delta);

	dlFullyConnectedLayer* m_pUpLayer;
	dlFullyConnectedLayer* m_pDownLayer;
	uint m_inputSize;
	uint m_outputSize;
	double* m_blas;
	double** m_weight;
	double* m_output;
	double m_rate;
	FunActivator m_Activator;

};