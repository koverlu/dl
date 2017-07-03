#pragma once
#include "dlLayer.h"

class dlFullyConnectedLayer
{
public:
	dlFullyConnectedLayer(uint inputSize, uint outputSize, FunActivator funActivator, 
		dlFullyConnectedLayer* pUpLayer, dlFullyConnectedLayer* pDownLayer);
	~dlFullyConnectedLayer();	
	void CalOutput();
	void SetInputData(MatrixXf& data);
	void UpdateWB(MatrixXf& delta);
	void CalDelta(MatrixXf& data, MatrixXf& dst_delta);
	void SetWB(MatrixXf& weight, MatrixXf& blas);
	dlFullyConnectedLayer* m_pUpLayer;
	dlFullyConnectedLayer* m_pDownLayer;
	uint m_inputSize;
	uint m_outputSize;
	MatrixXf m_blas;
	MatrixXf m_weight;	//m_weight(OUTPUT, INPUT)
	MatrixXf m_output;
	double m_rate;
	FunActivator m_Activator;

};