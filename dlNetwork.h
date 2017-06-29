#pragma once
#include "dlDef.h"

class dlNode;
class dlLayer;
class dlFullyConnectedLayer;

struct TRAIN_INFO
{
	uint trainTimes;
	uint testTimes;
	uint corrects;
	uint testCorrects;
	uint layerNum;
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
	void Train(MatrixXf& input, MatrixXf& target);
private:
	string m_name;
	TRAIN_INFO m_trainInfo;
	vector<dlFullyConnectedLayer*> m_layers;
	//void Accuracy();
};