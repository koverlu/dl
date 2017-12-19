#pragma once
#include "dlDef.h"
class FCLayer;
class LSTMLayer;
struct TRAIN_INFO
{
	uint trainTimes;
	uint faults;
	uint epoch;
	uint layerNum;
};

struct EPOCH_INFO
{
	uint times;
	uint faults;
};

class dlNetwork
{
public:
	dlNetwork(char * name, uint batchSize);
	~dlNetwork();
	void Init();
	void LoadInfo();
	void SaveInfo();
	double EpochStatistics();
	double GetLastErrorRate(uint num);
	//void CalGradient(MatrixXf& input, MatrixXf& target);
	void SetInputsAndTargets(vector<double>* pInputs, vector<double>* pTargets);
	void Train();
	void Test();
private:
	string m_name;
	TRAIN_INFO m_trainInfo;
	EPOCH_INFO m_epoch;
	vector<EPOCH_INFO> m_epochVector;
	FCLayer* m_pFCLayer;
	LSTMLayer* m_pLSTMLayer;
	uint m_thousandsFaults;
	uint m_batchSize;
};