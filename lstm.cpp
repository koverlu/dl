#include "lstm.h"
#include <math.h>
#include <string>
#include <assert.h>
double activ_sigmoid(double x)
{
	return (1 / (1 + exp(-x)));
}

double activ_tanh(double x)
{
	return tanh(x);
}

void VectorMul(const vector<float>& v_src1, 
			   const vector<float>& v_src2,  
			   vector<float>& v_dst, 
			   uint width, uint src1_offset = 0, uint src2_offset = 0, uint dst_offset = 0)
{
	assert(v_src1.size() >= src1_offset + width &&
		v_src2.size() >= src2_offset + width &&
		v_dst.size() >= dst_offset + width );
	for (int i = 0; i < width; i++)
		v_dst[dst_offset + i] = v_src1[src1_offset + i] * v_src2[src2_offset + i];
}

void VectorAdd(const vector<float>& v_src1, 
			   const vector<float>& v_src2,  
			   vector<float>& v_dst, 
			   uint width, uint src1_offset = 0, uint src2_offset = 0, uint dst_offset = 0)
{
	assert(v_src1.size() >= src1_offset + width &&
		v_src2.size() >= src2_offset + width &&
		v_dst.size() >= dst_offset + width );
	for (int i = 0; i < width; i++)
		v_dst[dst_offset + i] = v_src1[src1_offset + i] + v_src2[src2_offset + i];
}

void VectorMM(const vector<float>& v_src1, 
			  uint src1_rows,
			  uint src1_cols,
			  uint src1_offset,
			  const vector<float>& v_src2, 
			  uint src2_rows,
			  uint src2_cols,
			  uint src2_offset,
			  vector<float>& v_dst,
			  uint dst_offset)
{
	assert(src1_cols == src2_rows && 
		v_src1.size() >= (src1_offset + src1_rows * src1_cols) &&
		v_src2.size() >= (src2_offset + src2_rows * src2_cols) &&
		v_dst.size() >= (dst_offset + src1_rows * src2_cols));
	for(uint m = 0; m < src1_rows; m++)
		for (uint n = 0; n < src2_cols; n++)
			for (uint k = 0; k < src1_cols; k++)
				v_dst[dst_offset + m * src2_cols + n] += v_src1[src1_offset + m * src1_cols + k] * v_src2[src2_offset + k * src2_cols + n];
}

void VectorActive(vector<float>& v_dst, double(*activator)(double))
{
	for (uint i = 0; i < v_dst.size(); i++ )
		v_dst[i] = activator(v_dst[i]);
}

LSTMLayerNetWork::LSTMLayerNetWork(uint inVecLen, uint stateLen, uint batchSize, uint steps, float learnRate) :
m_inVecLen(inVecLen),
m_stateLen(stateLen),
m_batchSize(batchSize),
m_steps(steps),
m_learnRate(learnRate)
{
	uint totalStateLen = batchSize * stateLen;
	m_states.resize(totalStateLen);
	//bf, bi, bc, bo
	m_bias.resize(stateLen * 4);
	//Wf, Wi, Wc, Wo
	m_wtStride = m_stateLen * (m_stateLen + m_inVecLen);
	m_weights.resize(m_wtStride * 4);

}

void LSTMLayerNetWork::Forward()
{
	uint wtxoff = m_stateLen * m_stateLen;
	for(uint b =0; b < m_batchSize; b++)
	{
		for (uint t = 0; t < m_steps; t++)
		{
			uint input_offset = (b * m_batchSize + t) * m_inVecLen;
			uint state_offset = (b * m_batchSize + t) * m_stateLen;
			//Forget gate
			vector<float> ft(m_stateLen);
			memcpy(&ft[0], &m_bias[0], m_stateLen * sizeof(float));
			VectorMM(m_weights, m_stateLen, m_stateLen, 0, 
				m_states, m_stateLen, 1, 0, ft, 0);
			VectorMM(m_weights, m_stateLen, m_inVecLen, wtxoff,
				m_inputs, m_inVecLen, 1, input_offset, ft, 0);
			VectorActive(ft, activ_sigmoid);
			memcpy(&m_ft[state_offset], &ft[0], m_stateLen * sizeof(float));

			//Input gate
			vector<float> it(m_stateLen);
			memcpy(&it[0], &m_bias[m_stateLen], m_stateLen * sizeof(float));
			VectorMM(m_weights, m_stateLen, m_stateLen, m_wtStride, 
				m_states, m_stateLen, 1, 0, it, 0);
			VectorMM(m_weights, m_stateLen, m_inVecLen, m_wtStride + wtxoff,
				m_inputs, m_inVecLen, 1, input_offset, it, 0);
			VectorActive(it, activ_sigmoid);
			memcpy(&m_it[state_offset], &it[0], m_stateLen * sizeof(float));

			//Current state
			vector<float> ct(m_stateLen);
			memcpy(&ct[0], &m_bias[m_stateLen * 2], m_stateLen * sizeof(float));
			VectorMM(m_weights, m_stateLen, m_stateLen, m_wtStride * 2, 
				m_states, m_stateLen, 1, 0, ct, 0);
			VectorMM(m_weights, m_stateLen, m_inVecLen, m_wtStride * 2 + wtxoff,
				m_inputs, m_inVecLen, 1, input_offset, ct, 0);
			VectorActive(ct, activ_tanh);
			memcpy(&m_ct[state_offset], &ct[0], m_stateLen * sizeof(float));

			//Output state
			vector<float> ot(m_stateLen);
			memcpy(&ot[0], &m_bias[m_stateLen * 3], m_stateLen * sizeof(float));
			VectorMM(m_weights, m_stateLen, m_stateLen, m_wtStride * 3, 
				m_states, m_stateLen, 1, 0, ot, 0);
			VectorMM(m_weights, m_stateLen, m_inVecLen, m_wtStride * 3 + wtxoff,
				m_inputs, m_inVecLen, 1, input_offset, ot, 0);
			VectorActive(ot, activ_sigmoid);
			memcpy(&m_ot[state_offset], &ot[0], m_stateLen * sizeof(float));

			//
			vector<float> ftct(m_stateLen);
			VectorMul(ft, m_lstates, ftct, m_stateLen, 0, state_offset);
			vector<float> itct(m_stateLen);
			VectorMul(it, ct, itct, m_stateLen);
			VectorAdd(ftct, itct, m_lstates, m_stateLen, 0, 0, state_offset);

			vector<float> tanct(m_stateLen);
			memcpy(&tanct[0], &m_lstates[state_offset], m_stateLen * sizeof(float));		
			VectorActive(tanct, activ_tanh);
			VectorMul(ot, tanct, m_states, m_stateLen, 0, state_offset, state_offset);
		}
	}		
}

void LSTMLayerNetWork::BackWard()
{

}
