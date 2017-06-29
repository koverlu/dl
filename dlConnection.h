#pragma once
#include "dlDef.h"
class dlNode;
class dlLayer;
class dlConnection
{
public:
	vector<dlNode* >* m_pUpNodes;
	vector<dlNode* >* m_pDownNodes;
	dlLayer* m_upLayer;
	dlLayer* m_downLayer;
	bool** m_valids;
	double** m_weights;
	uint m_upNum;
	uint m_downNum;
};