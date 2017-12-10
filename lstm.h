#include <vector>
using namespace std;
typedef unsigned int uint;

class LSTMLayerNetWork
{
public:
	LSTMLayerNetWork(uint inVecLen, uint stateLen, uint batchSize, uint steps, float learnRate, const char* pInput = NULL);
	uint m_inVecLen;
	uint m_stateLen;
	uint m_batchSize;
	uint m_steps;
	uint m_wtStride;
	float m_learnRate;
	vector<float> m_states;
	vector<float> m_lstates;
	vector<float> m_lstates0;
	vector<float> m_inputs;
	vector<float> m_weights;
	vector<float> m_bias;
	vector<float> m_deltas;
	vector<float> m_ft;
	vector<float> m_it;
	vector<float> m_ct;
	vector<float> m_ot;
	vector<float> m_df;
	vector<float> m_di;
	vector<float> m_dc;
	vector<float> m_do;
	void GenerateInputs();
	void LoadInputsFromFile(const char* path);
	void DumpInputs();
	void Forward();
	void BackWard();
	void CalDelta();
	void CalGradient();
};