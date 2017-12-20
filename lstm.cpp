#include "lstm.h"
#include "common.h"

LSTMLayer::LSTMLayer(uint inVecLen, uint stateLen, uint batchSize, uint steps, double learnRate, const char* pInput) :
m_inVecLen(inVecLen),
m_stateLen(stateLen),
m_batchSize(batchSize),
m_steps(steps),
m_learnRate(learnRate),
m_pInputs(NULL),
m_bGenInputs(false)
{
	m_wtStride = m_stateLen * (m_stateLen + m_inVecLen);
	InitWeight();
	m_wei_grad.resize(m_wtStride * 4, 0);
	ResetStates();
	//if (pInput == NULL)
	//	GenerateInputs();
}

LSTMLayer::~LSTMLayer()
{
	if (m_bGenInputs && m_pInputs)
	{
		(*m_pInputs).clear();
		delete m_pInputs;
	}
}


void LSTMLayer::GenerateInputs()
{
	uint total_input_size = m_batchSize * m_steps * m_inVecLen;
	m_pInputs = new vector<double>;
	(*m_pInputs).resize(total_input_size);
	for (uint i = 0; i < total_input_size; i++)
		//(*m_pInputs)[i] = i+ 1;
		(*m_pInputs)[i] = rand() % 10;
	m_bGenInputs = true;
}

void LSTMLayer::Forward()
{
	ResetStates();
	uint wtxoff = m_stateLen * m_stateLen;
	for(uint b =0; b < m_batchSize; b++)
	{
		for (uint t = 0; t < m_steps; t++)
		{
			uint input_offset = (b * m_steps + t) * m_inVecLen;
			uint state_offset = (b * m_steps + t) * m_stateLen;
			uint offset_sub_one = t == 0 ? (b * m_stateLen) : (state_offset - m_stateLen);
			vector<double>& states = t == 0 ? m_states0 : m_states;

			//Forget gate
			vector<double> ft(m_stateLen);
			memcpy(&ft[0], &m_bias[0], m_stateLen * sizeof(double));
			VectorMM(m_weights, m_stateLen, m_stateLen, 0, 
				states, m_stateLen, 1, offset_sub_one, ft, 0);
			VectorMM(m_weights, m_stateLen, m_inVecLen, wtxoff,
				(*m_pInputs), m_inVecLen, 1, input_offset, ft, 0);
			VectorActive(ft, activ_sigmoid);
			memcpy(&m_ft[state_offset], &ft[0], m_stateLen * sizeof(double));

			//Input gate
			vector<double> it(m_stateLen);
			memcpy(&it[0], &m_bias[m_stateLen], m_stateLen * sizeof(double));
			VectorMM(m_weights, m_stateLen, m_stateLen, m_wtStride, 
				states, m_stateLen, 1, offset_sub_one, it, 0);
			VectorMM(m_weights, m_stateLen, m_inVecLen, m_wtStride + wtxoff,
				(*m_pInputs), m_inVecLen, 1, input_offset, it, 0);
			VectorActive(it, activ_sigmoid);
			memcpy(&m_it[state_offset], &it[0], m_stateLen * sizeof(double));

			//Current state
			vector<double> ct(m_stateLen);
			memcpy(&ct[0], &m_bias[m_stateLen * 2], m_stateLen * sizeof(double));
			VectorMM(m_weights, m_stateLen, m_stateLen, m_wtStride * 2, 
				states, m_stateLen, 1, offset_sub_one, ct, 0);
			VectorMM(m_weights, m_stateLen, m_inVecLen, m_wtStride * 2 + wtxoff,
				(*m_pInputs), m_inVecLen, 1, input_offset, ct, 0);
			VectorActive(ct, activ_tanh);
			memcpy(&m_ct[state_offset], &ct[0], m_stateLen * sizeof(double));

			//Output gate
			vector<double> ot(m_stateLen);
			memcpy(&ot[0], &m_bias[m_stateLen * 3], m_stateLen * sizeof(double));
			VectorMM(m_weights, m_stateLen, m_stateLen, m_wtStride * 3, 
				states, m_stateLen, 1, offset_sub_one, ot, 0);
			VectorMM(m_weights, m_stateLen, m_inVecLen, m_wtStride * 3 + wtxoff,
				(*m_pInputs), m_inVecLen, 1, input_offset, ot, 0);
			VectorActive(ot, activ_sigmoid);
			memcpy(&m_ot[state_offset], &ot[0], m_stateLen * sizeof(double));

			//Long term state
			vector<double> ftct(m_stateLen);
			vector<double>& lstates = t == 0 ? m_lstates0 : m_lstates;
			VectorMul(ft, lstates, ftct, m_stateLen, 0, offset_sub_one);
			vector<double> itct(m_stateLen);
			VectorMul(it, ct, itct, m_stateLen);			
			VectorAdd(ftct, itct, m_lstates, m_stateLen, 0, 0, state_offset);

			//Output sate
			vector<double> tanhct(m_stateLen);
			memcpy(&tanhct[0], &m_lstates[state_offset], m_stateLen * sizeof(double));		
			VectorActive(tanhct, activ_tanh);
			VectorMul(ot, tanhct, m_states, m_stateLen, 0, 0, state_offset);
		}
		memcpy(&m_output[b * m_stateLen], &m_states[(b * m_steps + m_steps - 1) * m_stateLen], m_stateLen * sizeof(double));
	}		
}

void LSTMLayer::BackWard()
{
	CalDelta();
	CalGradient();
	UpdateWeights();
}

void LSTMLayer::CalDelta()
{
	
	for (uint b = 0; b < m_batchSize; b++)
	{
		memcpy(&m_deltas[(b * m_steps + m_steps - 1) * m_stateLen], &(*m_pBackDeltas)[b * m_stateLen], m_stateLen * sizeof(double));
		for (int t = m_steps - 1; t >= 0; t--)
		{
			//
			uint state_offset = (b * m_steps + t) * m_stateLen;

			//tanh(ct)			
			vector<double> tanhct(m_stateLen);
			memcpy(&tanhct[0], &m_lstates[state_offset], m_stateLen * sizeof(double));
			VectorActive(tanhct, activ_tanh);

			//delta_ot
			vector<double> tmp(m_stateLen);
			VectorOneSub(m_ot, tmp, m_stateLen, state_offset, 0);
			vector<double> mul_tmp(m_stateLen);
			VectorMul(m_deltas, tanhct, mul_tmp, m_stateLen, state_offset, 0, 0);
			VectorMul(mul_tmp, m_ot, mul_tmp, m_stateLen, 0, state_offset, 0);
			VectorMul(mul_tmp, tmp, m_do, m_stateLen, 0, 0, state_offset);

			//tmp = 1 - tanh(ct) * 	tanh(ct)		
			VectorMul(tanhct, tanhct, tmp);
			VectorOneSub(tmp, tmp);
			VectorMul(m_deltas, m_ot, mul_tmp, m_stateLen, state_offset, state_offset, 0);
			//delta_ot_tanh = delta_t * ot * (1 - tanh(ct) * 	tanh(ct))
			vector<double> delta_ot_tanh(m_stateLen);
			VectorMul(mul_tmp, tmp, delta_ot_tanh);

			//delta_ft
			//tmp = 1 - ft
			VectorOneSub(m_ft, tmp, m_stateLen, state_offset, 0);
			uint offset_sub_one = t == 0 ? (b * m_stateLen) : (state_offset - m_stateLen);
			vector<double>& lstates = t == 0 ? m_lstates0 : m_lstates;
			VectorMul(delta_ot_tanh, lstates, mul_tmp, m_stateLen, 0, offset_sub_one, 0);
			VectorMul(mul_tmp, m_ft, mul_tmp, m_stateLen, 0, state_offset, 0);
			VectorMul(mul_tmp, tmp, m_df, m_stateLen, 0, 0, state_offset);

			//delta_it
			//tmp = 1 - it
			VectorOneSub(m_it, tmp, m_stateLen, state_offset, 0);
			VectorMul(delta_ot_tanh, m_ct, mul_tmp, m_stateLen, 0, state_offset, 0);
			VectorMul(mul_tmp, m_it, mul_tmp, m_stateLen, 0, state_offset, 0);
			VectorMul(mul_tmp, tmp, m_di, m_stateLen, 0, 0, state_offset);

			//delta_ct
			//tmp = 1 - ct * ct
			VectorMul(m_ct, m_ct, tmp, m_stateLen, state_offset, state_offset, 0);
			VectorOneSub(tmp, tmp);
			VectorMul(delta_ot_tanh, m_it, mul_tmp, m_stateLen, 0, state_offset, 0);
			VectorMul(mul_tmp, tmp, m_dc, m_stateLen, 0, 0, state_offset);

			//delta_t-1
			if(t > 0)
			{
				VectorMM(m_do, 1, m_stateLen, state_offset, m_weights, m_stateLen, m_stateLen, m_wtStride * 3, m_deltas, state_offset - m_stateLen);
				VectorMM(m_df, 1, m_stateLen, state_offset, m_weights, m_stateLen, m_stateLen, 0, m_deltas, state_offset - m_stateLen);
				VectorMM(m_di, 1, m_stateLen, state_offset, m_weights, m_stateLen, m_stateLen, m_wtStride * 1, m_deltas, state_offset - m_stateLen);
				VectorMM(m_dc, 1, m_stateLen, state_offset, m_weights, m_stateLen, m_stateLen, m_wtStride * 2, m_deltas, state_offset - m_stateLen);

			}
		}
	}
}

void LSTMLayer::CalGradient()
{
	vector<double> sum_wei_grad(m_wtStride * 4, 0);
	vector<double> sum_bias_grad(m_stateLen * 4, 0);
	for (uint b = 0; b < m_batchSize; b++)
	{
		for (uint t = 0; t < m_steps; t++)
		{
			uint state_offset = (b * m_steps + t) * m_stateLen;
			uint offset_sub_one = t == 0 ? (b * m_stateLen) : (state_offset - m_stateLen);
			vector<double>& states = t == 0 ? m_states0 : m_states;
			VectorMM(m_do, m_stateLen, 1, state_offset, states, 1, m_stateLen, offset_sub_one, sum_wei_grad, m_wtStride * 3);
			VectorMM(m_df, m_stateLen, 1, state_offset, states, 1, m_stateLen, offset_sub_one, sum_wei_grad, 0);
			VectorMM(m_di, m_stateLen, 1, state_offset, states, 1, m_stateLen, offset_sub_one, sum_wei_grad, m_wtStride * 1);
			VectorMM(m_dc, m_stateLen, 1, state_offset, states, 1, m_stateLen, offset_sub_one, sum_wei_grad, m_wtStride * 2);
			//
			VectorAdd(m_do, sum_bias_grad, sum_bias_grad, m_stateLen, 0, m_stateLen * 3, m_stateLen * 3);
			VectorAdd(m_df, sum_bias_grad, sum_bias_grad, m_stateLen, 0, 0, 0);
			VectorAdd(m_di, sum_bias_grad, sum_bias_grad, m_stateLen, 0, m_stateLen * 1, m_stateLen * 1);
			VectorAdd(m_dc, sum_bias_grad, sum_bias_grad, m_stateLen, 0, m_stateLen * 2, m_stateLen * 2);
		}
		//Calculate 
		uint last_offset = (b * m_steps + m_steps - 1)  * m_stateLen;
		uint input_offset = (b * m_steps + m_steps - 1) * m_inVecLen;
		uint wtxoff = m_stateLen * m_stateLen;
		VectorMM(m_do, m_stateLen, 1, last_offset, (*m_pInputs), 1, m_inVecLen, input_offset, sum_wei_grad, m_wtStride * 3 + wtxoff);
		VectorMM(m_df, m_stateLen, 1, last_offset, (*m_pInputs), 1, m_inVecLen, input_offset, sum_wei_grad, wtxoff);
		VectorMM(m_di, m_stateLen, 1, last_offset, (*m_pInputs), 1, m_inVecLen, input_offset, sum_wei_grad, m_wtStride * 1 + wtxoff);
		VectorMM(m_dc, m_stateLen, 1, last_offset, (*m_pInputs), 1, m_inVecLen, input_offset, sum_wei_grad, m_wtStride * 2 + wtxoff);
	}
	m_wei_grad = sum_wei_grad;
	m_bias_grad = sum_bias_grad;
}

void LSTMLayer::GradientCheck()
{
	//batchSize = 1
	double epsilon = 0.0001;
	uint last_state_offset = (m_steps - 1) * m_stateLen;
	Forward();
	for (uint i = 0; i < m_stateLen; i++)
		//m_deltas[last_state_offset + i] = 1.0;
		m_deltas[last_state_offset + i] =  m_states[last_state_offset + i];
	BackWard();
	
	for (uint i = 0; i < m_stateLen; i++)
	{
		for (uint j = 0; j < m_stateLen + m_inVecLen; j++)
		{
			ResetStates();
			m_weights[i * (m_stateLen + m_inVecLen) + j] += epsilon;
			Forward();
			//tmp = ht * ht
			vector<double> tmp(m_stateLen);
			VectorMul(m_states, m_states, tmp, m_stateLen, last_state_offset, last_state_offset, 0);
			double err1 = VectorSum(tmp, m_stateLen, 0) / 2.0;
			//double err1 = VectorSum(m_states, m_stateLen, last_state_offset);
			ResetStates();
			m_weights[i * (m_stateLen + m_inVecLen) + j] -= 2 * epsilon;
			Forward();
			VectorMul(m_states, m_states, tmp, m_stateLen, last_state_offset, last_state_offset, 0);
			double err2 = VectorSum(tmp, m_stateLen, 0) / 2.0;
			//double err2 = VectorSum(m_states, m_stateLen, last_state_offset);
			double expect_grad = (err1 - err2) / (2 * epsilon);
			m_weights[i * (m_stateLen + m_inVecLen) + j] += epsilon;
			printf("weights(%d,%d): expected - actural %.4e - %.4e\n", i, j, expect_grad, m_wei_grad[i * (m_stateLen + m_inVecLen) + j]);
		}
	}
}

void LSTMLayer::ResetStates()
{
	uint totalStateLen = m_batchSize * m_stateLen * m_steps;
	VectorResizeZero(m_states, totalStateLen);
	VectorResizeZero(m_states0, m_batchSize * m_stateLen);
	VectorResizeZero(m_lstates, totalStateLen);
	VectorResizeZero(m_lstates0, m_batchSize * m_stateLen);	
	VectorResizeZero(m_output, m_batchSize * m_stateLen);
	VectorResizeZero(m_deltas, totalStateLen);
	VectorResizeZero(m_ft, totalStateLen);
	VectorResizeZero(m_it, totalStateLen);
	VectorResizeZero(m_ct, totalStateLen);
	VectorResizeZero(m_ot, totalStateLen);
	VectorResizeZero(m_df, totalStateLen);
	VectorResizeZero(m_di, totalStateLen);
	VectorResizeZero(m_dc, totalStateLen);
	VectorResizeZero(m_do, totalStateLen);
}

void LSTMLayer::InitWeight()
{
	//bf, bi, bc, bo
	VectorResizeZero(m_bias, m_stateLen * 4);
	srand(2);
	//Wf, Wi, Wc, Wo
	m_weights.resize(m_wtStride * 4);
	for (uint i = 0; i < m_weights.size(); i++)
		//m_weights[i] = 1.0/10000.0;
		m_weights[i] = rand() / ((double)RAND_MAX / 2.0) - 1.0;
}

void LSTMLayer::UpdateWeights()
{
	//vector<double> avg_factor(m_wtStride * 4, -m_learnRate / m_batchSize);
	vector<double> avg_factor(m_wtStride * 4, -m_learnRate);
	VectorMul(m_wei_grad, avg_factor, m_wei_grad);
	VectorAdd(m_weights, m_wei_grad, m_weights);

	VectorMul(m_bias_grad, avg_factor, m_bias_grad);
	VectorAdd(m_bias, m_bias_grad, m_bias);
}

void LSTMLayer::SetConnection(vector<double>* pInputs, vector<double>* pDeltas)
{
	
}