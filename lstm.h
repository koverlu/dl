#include <vector>
using namespace std;
typedef unsigned int uint;

class LSTMLayerNetWork
{
public:
	LSTMLayerNetWork(uint inVecLen, uint stateLen, uint batchSize, uint steps, double learnRate, const char* pInput = NULL);
	uint m_inVecLen;
	uint m_stateLen;
	uint m_batchSize;
	uint m_steps;
	uint m_wtStride;
	double m_learnRate;
	vector<double> m_states;
	vector<double> m_states0;	//t = -1
	vector<double> m_lstates;
	vector<double> m_lstates0;	//t = -1
	vector<double> m_inputs;
	vector<double> m_weights;
	vector<double> m_wei_grad;
	vector<double> m_bias;
	vector<double> m_deltas;
	vector<double> m_ft;
	vector<double> m_it;
	vector<double> m_ct;
	vector<double> m_ot;
	vector<double> m_df;
	vector<double> m_di;
	vector<double> m_dc;
	vector<double> m_do;
	void ResetStates();
	void GenerateInputs();
	void LoadInputsFromFile(const char* path);
	void DumpInputs();
	void Forward();
	void BackWard();
	void CalDelta();
	void CalGradient();
	void GradientCheck();
	void InitWeight();
};