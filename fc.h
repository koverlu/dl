#include "dlDef.h"

class FCLayer
{
public:
	FCLayer(uint inVecLen, uint stateLen, uint batchSize, double learnRate, const char* pInput = NULL);
	~FCLayer();
	uint m_inVecLen;
	uint m_stateLen;
	uint m_batchSize;
	double m_learnRate;
	uint m_totalStateLen;
	bool m_bGenInputs;
	vector<double> m_states;
	vector<double> m_weights;
	vector<double> m_bias;
	vector<double> m_deltas;
	vector<double>* m_pInputs;
	vector<double>* m_pTargets;
	vector<double> m_wei_grad;
	vector<double> m_bias_grad;
	vector<double> m_back_deltas;
	void Forward();
	void BackWard();
	void CalDelta();
	void CalGradient();
	void UpdateWeights();
	void GradientCheck();
	void InitWeight();
	void SetConnection(vector<double>* pInputs, vector<double>* pTargets);
};