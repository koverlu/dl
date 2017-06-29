#pragma once
#include <vector>
#include "dlDef.h"

class dlNode;
class dlNetwork;
class dlConnection;
enum dlLayerType
{
	DL_INPUT,
	DL_HIDDEN,
	DL_OUTPUT
};

class dlLayer
{
public:
	dlLayer(dlLayerType type, uint32_t num);
	void Init();
private:
	dlNetwork* m_pNW;
	dlLayerType m_type;
	uint32_t m_numNode;
	vector<dlNode*> m_nodes;
	uint32_t m_layerId;
};