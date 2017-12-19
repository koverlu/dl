#include "fc.h"
#include "common.h"
#include <math.h>

FCLayer::FCLayer(uint inVecLen, uint stateLen, uint batchSize, double learnRate, const char* pInput /*= NULL*/) :
m_inVecLen(inVecLen),
m_stateLen(stateLen),
m_batchSize(batchSize),
m_learnRate(learnRate),
m_bGenInputs(false)
{
	m_totalStateLen = m_batchSize * m_stateLen;
	VectorResizeZero(m_deltas, m_totalStateLen);
	VectorResizeZero(m_states, m_totalStateLen);
	InitWeight();
	//m_pInputs = new vector<double>;
	//VectorResizeZero(*m_pInputs, m_inVecLen * m_batchSize);
	//for (uint i = 0; i < m_inVecLen * m_batchSize; i++)
	//	(*m_pInputs)[i] = i;
}


FCLayer::~FCLayer()
{
	if (m_bGenInputs && m_pInputs)
	{
		(*m_pInputs).clear();
		delete m_pInputs;
	}
}

void FCLayer::InitWeight()
{
	m_bias.resize(m_stateLen, 0);
	srand(2);
	m_weights.resize(m_stateLen * m_inVecLen);
	for (uint i = 0; i < m_weights.size(); i++)
		m_weights[i] = rand() / ((double)RAND_MAX / 2.0) - 1.0;
}

void FCLayer::Forward()
{
	for(uint b =0; b < m_batchSize; b++)
	{
		uint state_offset = b * m_stateLen;
		memcpy(&m_states[state_offset], &m_bias[0], m_stateLen * sizeof(double));
		VectorMM(m_weights, m_stateLen, m_inVecLen, 0, 
			*m_pInputs, m_inVecLen, 1, b * m_inVecLen, m_states, state_offset);
		VectorSoftmax(m_states, m_stateLen, state_offset);
	}
}

void FCLayer::BackWard()
{
	CalDelta();
	CalGradient();
	UpdateWeights();
}

void FCLayer::CalDelta()
{	

	VectorResizeZero(m_back_deltas, m_batchSize * m_inVecLen);
	for(uint b =0; b < m_batchSize; b++)
	{
		//Softmax delta
		for (uint i = 0; i < m_stateLen; i++)
		{
			uint offset = b * m_stateLen + i;
			if (m_pTargets->at(offset) == 0)
				m_deltas[offset] = m_states[offset];
			else
				m_deltas[offset] = m_states[offset] - 1.0;
		}

		VectorMM(m_deltas, 1, m_stateLen, b * m_stateLen,
			m_weights, m_stateLen, m_inVecLen, 0, m_back_deltas, b * m_inVecLen);
	}
}

void FCLayer::CalGradient()
{
	vector<double> sum_wei_grad(m_stateLen * m_inVecLen, 0);
	vector<double> sum_bias_grad(m_stateLen, 0);
	for (uint b = 0; b < m_batchSize; b++)
	{
		{
			VectorMM(m_deltas, m_stateLen, 1, b * m_stateLen, *m_pInputs, 1, m_inVecLen, b * m_inVecLen, sum_wei_grad, 0);
			VectorAdd(m_deltas, sum_bias_grad, sum_bias_grad, m_stateLen, b * m_stateLen, 0, 0);
		}
	}
	m_wei_grad = sum_wei_grad;
	m_bias_grad = sum_bias_grad;
}

void FCLayer::GradientCheck()
{
	double epsilon = 0.0001;
	Forward();
	m_pTargets = new vector<double>;
	VectorResizeZero(*m_pTargets, m_stateLen * m_batchSize);
	vector<uint> v_rand1(m_batchSize);
	for (uint i = 0; i < m_batchSize; i++)
	{
		v_rand1[i] = rand() % m_stateLen;
		(*m_pTargets)[i * m_stateLen + v_rand1[i]] = 1;
	}
		
	BackWard();
	for (uint i = 0; i < m_stateLen; i++)
	{
		for (uint j = 0; j < m_stateLen; j++)
		{
			m_weights[i * m_stateLen + j] += epsilon;
			Forward();
			double err1 = -log(m_states[i * m_stateLen + v_rand1[i]]);
			m_weights[i * m_stateLen + j] -= 2 * epsilon;
			Forward();
			double err2 = -log(m_states[i * m_stateLen + v_rand1[i]]);
			double expect_grad = (err1 - err2) / (2 * epsilon);
			m_weights[i * m_stateLen + j] += epsilon;
			printf("weights(%d,%d): expected - actural %.4e - %.4e\n", i, j, expect_grad, m_wei_grad[i * m_stateLen + j]);
		}
	}
}

void FCLayer::UpdateWeights()
{
	vector<double> avg_factor(m_stateLen * m_inVecLen, -m_learnRate / m_batchSize);
	VectorMul(m_wei_grad, avg_factor, m_wei_grad);
	VectorAdd(m_weights, m_wei_grad, m_weights);
	
	VectorMul(m_bias_grad, avg_factor, m_bias_grad);
	VectorAdd(m_bias, m_bias_grad, m_bias);
}

void FCLayer::SetConnection(vector<double>* pInputs, vector<double>* pTargets)
{
	m_pInputs = pInputs;
	m_pTargets = pTargets;
}


