#include "ce.h"
#include "dlDef.h"
#include "dlNetwork.h"
#include "dlFullyConnectedLayer.h"
#include <fstream>
//#include <iostream>
extern void TestMain();

struct MNISTLabelFileHeader
{
	uchar MagicNumber[4];
	uchar NumberOfLabels[4];
};

struct MNISTImageFileHeader
{
	uchar MagicNumber[4];
	uchar NumberOfImages[4];
	uchar NumberOfRows[4];
	uchar NumberOfColums[4];
};

int ConvertCharArrayToInt(unsigned char* array, int LengthOfArray)
{
	if (LengthOfArray < 0)
	{
		return -1;
	}
	int result = static_cast<signed int>(array[0]);
	for (int i = 1; i < LengthOfArray; i++)
	{
		result = (result << 8) + array[i];
	}
	return result;
}

void main()
{
	dlNetwork mnist("mnist");
	//mnist.Init();
	ifstream file("train-labels.idx1-ubyte", ios::binary);
	MNISTLabelFileHeader header;
	file.read((char*)&header, sizeof(MNISTLabelFileHeader));
	int magicNum = ConvertCharArrayToInt(header.MagicNumber, 4);
	DBG_ASSERT(magicNum == 0x00000801, "Magic num is incorrect!");
	int times = ConvertCharArrayToInt(header.NumberOfLabels, 4);
	uchar* input = new uchar[times];
	file.read((char*)input, times);
	file.close();

	ifstream imgFile("train-images.idx3-ubyte", ios::binary);
	MNISTImageFileHeader imgHeader;
	imgFile.read((char*)&imgHeader, sizeof(MNISTImageFileHeader));
	magicNum = ConvertCharArrayToInt(imgHeader.MagicNumber, 4);
	DBG_ASSERT(magicNum == 0x00000803, "Magic num is incorrect!");
	times = ConvertCharArrayToInt(imgHeader.NumberOfImages, 4);
	int row = ConvertCharArrayToInt(imgHeader.NumberOfRows, 4);
	int col = ConvertCharArrayToInt(imgHeader.NumberOfColums, 4);
	uchar* inputImg = new uchar[row * col * times];
	imgFile.read((char*)inputImg, row * col * times);
	imgFile.close();
	mnist.Init();

	float* inputData = new float[row * col];
	char at[10];
	uint CPUTIME_START = GetTickCount();
	for (uint i = 0; i < times; i++)
	{
		for (uint j = 0; j < row * col; j++)
		{
			inputData[j] = (float)(*(inputImg + i * row * col + j));
		}
		Map<MatrixXf> inputMap(inputData, row * col, 1);
		MatrixXf inputM = inputMap;
		MatrixXf targatM = MatrixXf::Zero(10, 1);
		targatM(input[i], 0) = 1;
		mnist.Train(inputM, targatM);

		if (i%100 == 0)
		{
			memset(at, ' ', 10);
			at[input[i]] = '@';
			MatrixXf outputM = (mnist.GetLayer(2))->m_output;
			DBG_PRINT("%c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f\n",
				at[0], outputM(0, 0), at[1], outputM(1, 0), at[2], outputM(2, 0), at[3], outputM(3, 0), at[4], outputM(4, 0),
				at[5], outputM(5, 0), at[6], outputM(6, 0), at[7], outputM(7, 0), at[8], outputM(8, 0), at[9], outputM(9, 0));
		}
	}
	uint CPUTIME_END = GetTickCount();
	DBG_PRINT("CPU TIME: %d Seconds\n", (CPUTIME_END - CPUTIME_START) / 1000);
	mnist.SaveInfo();
	delete[] input;
	delete[] inputImg;
}