#include <assert.h>
#include <math.h>
#include "dlDef.h"

double activ_sigmoid(double x)
{
	return (1 / (1 + exp(-x)));
}

double activ_tanh(double x)
{
	return tanh(x);
}

double VectorSum(const vector<double>& v_src, uint width = 0, uint offset = 0)
{
	assert(v_src.size() >= width + offset);
	double sum = 0;
	for (int i = offset; i < offset + width; i++)
		sum += v_src[i];
	return sum;
}

void VectorMul(vector<double>& v_src1, 
			   vector<double>& v_src2,  
			   vector<double>& v_dst, 
			   uint width = 0, uint src1_offset = 0, uint src2_offset = 0, uint dst_offset = 0)
{
	if (width == 0)
		width = v_src1.size();
	assert(v_src1.size() >= src1_offset + width &&
		v_src2.size() >= src2_offset + width &&
		v_dst.size() >= dst_offset + width );
	for (int i = 0; i < width; i++)
		v_dst[dst_offset + i] = v_src1[src1_offset + i] * v_src2[src2_offset + i];
}

void VectorAdd(vector<double>& v_src1, 
			   vector<double>& v_src2,  
			   vector<double>& v_dst, 
			   uint width = 0, uint src1_offset = 0, uint src2_offset = 0, uint dst_offset = 0)
{
	if (width == 0)
		width = v_src1.size();
	assert(v_src1.size() >= src1_offset + width &&
		v_src2.size() >= src2_offset + width &&
		v_dst.size() >= dst_offset + width );
	for (int i = 0; i < width; i++)
		v_dst[dst_offset + i] = v_src1[src1_offset + i] + v_src2[src2_offset + i];
}

void VectorResizeZero(vector<double>& v_dst, uint size)
{
	v_dst.resize(size);
	memset(&v_dst[0], 0, size * sizeof(double));
}

//(1 - v)
void VectorOneSub(vector<double>& v_src,
				  vector<double>& v_dst,
				  uint width = 0, uint src_offset = 0, uint dst_offset = 0)
{
	if (width == 0)
		width = v_src.size();
	assert(v_src.size() >= src_offset + width &&
		v_dst.size() >= dst_offset + width);
	for (int i = 0; i < width; i++)
		v_dst[dst_offset + i] = 1 - v_src[src_offset + i];
}

void VectorMM(const vector<double>& v_src1, 
			  uint src1_rows,
			  uint src1_cols,
			  uint src1_offset,
			  const vector<double>& v_src2, 
			  uint src2_rows,
			  uint src2_cols,
			  uint src2_offset,
			  vector<double>& v_dst,
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

void VectorActive(vector<double>& v_dst, double(*activator)(double))
{
	for (uint i = 0; i < v_dst.size(); i++ )
		v_dst[i] = activator(v_dst[i]);
}

void VectorSoftmax(vector<double>& v_dst, uint len /*= 0*/, uint offset /*= 0*/)
{
	if (len == 0)
		len = v_dst.size();
	vector<double> exp_value(len);
	double sum = 0;
	for (uint i = 0; i < len; i++)
	{
		exp_value[i] = exp(v_dst[offset + i]);
		sum += exp_value[i];
	}
	for (uint i = 0; i < len; i++)
	{
		v_dst[offset + i] = exp_value[i] / sum;
	}
}

uint VectorMaxIdx(const vector<double>& v_src, uint len /*= 0*/, uint offset /*= 0*/)
{
	double max_value = -1.0;
	uint max_idx = 0;
	if (len == 0)
		len = v_src.size();
	for (uint i = 0; i < len; i++)
	{
		if (v_src[offset + i] >= max_value)
		{
			max_value = v_src[offset + i];
			max_idx = i;
		}
	}
	return max_idx;
}
