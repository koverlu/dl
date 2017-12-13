#include "dlDef.h"

extern double activ_sigmoid(double x);
extern double activ_tanh(double x);

extern void VectorResizeZero(vector<double>& v_dst, uint size);

extern double VectorSum(const vector<double>& v_src, uint width = 0, uint offset = 0);

extern void VectorMul(vector<double>& v_src1, 
			   vector<double>& v_src2,  
			   vector<double>& v_dst, 
			   uint width = 0, uint src1_offset = 0, uint src2_offset = 0, uint dst_offset = 0);

extern void VectorAdd(vector<double>& v_src1, 
			   vector<double>& v_src2,  
			   vector<double>& v_dst, 
			   uint width = 0, uint src1_offset = 0, uint src2_offset = 0, uint dst_offset = 0);

extern void VectorOneSub(vector<double>& v_src,
				  vector<double>& v_dst,
				  uint width = 0, uint src_offset = 0, uint dst_offset = 0);

extern void VectorMM(const vector<double>& v_src1, 
			  uint src1_rows,
			  uint src1_cols,
			  uint src1_offset,
			  const vector<double>& v_src2, 
			  uint src2_rows,
			  uint src2_cols,
			  uint src2_offset,
			  vector<double>& v_dst,
			  uint dst_offset);
extern void VectorActive(vector<double>& v_dst, double(*activator)(double));
extern void VectorSoftmax(vector<double>& v_dst, uint len = 0, uint offset = 0);