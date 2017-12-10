#include "lstm.h"

void main()
{
	LSTMLayerNetWork* pLSTM = new LSTMLayerNetWork(28, 128, 10, 30, 0.0001);
	pLSTM->Forward();
}