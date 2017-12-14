#include "dlDef.h"
#include "lstm.h"
#include "fc.h"
#include "network.h"
#include <fstream>
#include <iostream>
#include "debug.h"

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

void train()
{
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

	dlNetwork mnist("mnist");
	//Create Network, LSTM + FC
	mnist.Init();
	//Create Input Buffer and Target Buffer
	vector<double> inputData(row * col);
	vector<double> targetData(10);
	mnist.SetInputsAndTargets(&inputData, &targetData);

	uint CPUTIME_START = GetTickCount();
	uint epoch = 0;
	double errorRate = 0;
	double lastErrorRate = 1.0f;
	while (1)
	{
		for (uint i = 0; i < times; i++)
		{
			for (uint j = 0; j < row * col; j++)
			{
				inputData[j] = (float)(*(inputImg + i * row * col + j));
			}
			memset(&targetData[0], 0, 10 * sizeof(double));
			targetData[input[i]] = 1;
			
			mnist.Train();
			if (i % 100 == 0)
			{
				//memset(at, ' ', 10);
				//at[input[i]] = '@';
				//MatrixXf outputM = (mnist.GetLayer(2))->m_output;
				//DBG_PRINT("%c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f %c%.2f\n",
				//	at[0], outputM(0, 0), at[1], outputM(1, 0), at[2], outputM(2, 0), at[3], outputM(3, 0), at[4], outputM(4, 0),
				//	at[5], outputM(5, 0), at[6], outputM(6, 0), at[7], outputM(7, 0), at[8], outputM(8, 0), at[9], outputM(9, 0));
			}
		}
		mnist.EpochStatistics();
		if (epoch % 10 == 0)
		{
			errorRate = 0;
			errorRate = mnist.GetLastErrorRate(10);
			if (errorRate > lastErrorRate)
				break;
			else
				lastErrorRate = errorRate;
		}
		epoch++;
	}
	uint CPUTIME_END = GetTickCount();
	DBG_PRINT("CPU TIME: %d Seconds\n", (CPUTIME_END - CPUTIME_START) / 1000);
	delete[] input;
	delete[] inputImg;
}

void test()
{
	ifstream file("t10k-labels.idx1-ubyte", ios::binary);
	MNISTLabelFileHeader header;
	file.read((char*)&header, sizeof(MNISTLabelFileHeader));
	int magicNum = ConvertCharArrayToInt(header.MagicNumber, 4);
	DBG_ASSERT(magicNum == 0x00000801, "Magic num is incorrect!");
	int times = ConvertCharArrayToInt(header.NumberOfLabels, 4);
	uchar* input = new uchar[times];
	file.read((char*)input, times);
	file.close();

	ifstream imgFile("t10k-images.idx3-ubyte", ios::binary);
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

	//mnist.LoadInfo();

	float* inputData = new float[row * col];
	for (uint i = 0; i < times; i++)
	{
		for (uint j = 0; j < row * col; j++)
		{
			inputData[j] = (float)(*(inputImg + i * row * col + j));
		}
	}

	DBG_PRINT("%f\n", 1.0);
	delete[] input;
	delete[] inputImg;
}

void main(int argc, char * argv[])
{
	if (argc == 2)
	{
		string str = argv[1];
		if (str == "t")
		{
			train();
		}
		else
			DBG_PRINT("Wrong argument!\n");
	}
	else if (argc == 1)
	{
		test();
	}
	else
		DBG_PRINT("Wrong argument!\n");	
}