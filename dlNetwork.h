#pragma once
#include "dlDef.h"

class dlNode;
class dlLayer;
class dlFullyConnectedLayer;

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
	dlNetwork(char* name);
	~dlNetwork();
	dlFullyConnectedLayer* GetLayer(uint id);
	void Init();
	void LoadInfo();
	void SaveInfo();
	double EpochStatistics();
	double GetLastErrorRate(uint num);
	void CalGradient(MatrixXf& input, MatrixXf& target);
	void Train(MatrixXf& input, MatrixXf& target);
	void Test(MatrixXf& input, MatrixXf& target);
private:
	string m_name;
	TRAIN_INFO m_trainInfo;
	vector<dlFullyConnectedLayer*> m_layers;
	EPOCH_INFO m_epoch;
	vector<EPOCH_INFO> m_epochVector;
	//void Accuracy();
};