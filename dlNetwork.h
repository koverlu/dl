#pragma once
#include "dlDef.h"

class dlNode;
class dlLayer;
class dlFullyConnectedLayer;
class dlNetwork
{
public:
	dlLayer* GetLayer(uint id);
	void Init();
private:
	vector<dlFullyConnectedLayer*> m_layers;
};