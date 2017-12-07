#include <vector>
using namespace std;

class LSTMLayerNetWork
{
	uint m_inVecLen;
	uint m_stateLen;
	uint m_batchSize;
	uint m_steps;
	vector<float> m_state;
	vector<float> m_input;
	void GenerateInputs();
	void LoadInputsFromFile(const char* path);
	void DumpInputs();
	void Forward();
	void BackWard();
};