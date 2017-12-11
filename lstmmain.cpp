#include "lstm.h"

void main()
{
	LSTMLayerNetWork* pLSTM = new LSTMLayerNetWork(28, 128, 1, 30, 0.001);
	pLSTM->Forward();
	pLSTM->BackWard();
}