#pragma once
#include "dlDef.h"
class dlNetwork;

class dlNode
{

	vector<dlNode*> m_upStream;
	vector<dlNode*> m_downStream;
	dlNetwork* m_pNW;
	uint32_t m_nodeId;
	uint32_t m_layerId;
	double m_output;
	double m_delta;
	void CalOutput();
};