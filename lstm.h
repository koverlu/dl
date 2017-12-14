#include "dlDef.h"

class LSTMLayer
{
public:
	LSTMLayer(uint inVecLen, uint stateLen, uint batchSize, uint steps, double learnRate, const char* pInput = NULL);
	~LSTMLayer();
	uint m_inVecLen;
	uint m_stateLen;
	uint m_batchSize;
	uint m_steps;
	uint m_wtStride;
	double m_learnRate;
	bool m_bGenInputs;
	vector<double> m_states;
	vector<double> m_states0;	//t = -1
	vector<double> m_lstates;
	vector<double> m_lstates0;	//t = -1
	vector<double>* m_pInputs;
	vector<double> m_weights;
	vector<double> m_wei_grad;
	vector<double> m_bias_grad;
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
	vector<double>* m_back_deltas;
	vector<double> m_output;
	void ResetStates();
	void GenerateInputs();
	void LoadInputsFromFile(const char* path);
	void DumpInputs();
	void Forward();
	void BackWard();
	void CalDelta();
	void CalGradient();
	void UpdateWeights();
	void GradientCheck();
	void InitWeight();
	void SetConnection(vector<double>* pInputs, vector<double>* pDeltas);
};