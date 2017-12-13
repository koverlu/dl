#include "lstm.h"
#include "fc.h"

void lstmmain()
{
	//LSTMLayer* pLSTM = new LSTMLayer(28, 128, 1, 2, 0.001);
	//pLSTM->GradientCheck();
	//pLSTM->Forward();
	//pLSTM->BackWard();
	FCLayer* pFC = new FCLayer(128, 10, 1, 0.001);
	pFC->GradientCheck();
}