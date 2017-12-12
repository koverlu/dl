#include "lstm.h"

void main()
{
	LSTMLayerNetWork* pLSTM = new LSTMLayerNetWork(3, 2, 1, 2, 0.001);
	pLSTM->GradientCheck();
	//pLSTM->Forward();
	//pLSTM->BackWard();
}