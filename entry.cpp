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

void train()
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
			Map<MatrixXf> inputMap(inputData, row * col, 1);
			MatrixXf inputM = inputMap;
			MatrixXf targatM = MatrixXf::Ones(10, 1);
			targatM = targatM * 0.1;
			targatM(input[i], 0) = 0.9;
			mnist.CalGradient(inputM, targatM);
			mnist.Train(inputM, targatM);

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
	mnist.SaveInfo();
	delete[] input;
	delete[] inputImg;
}

void test()
{
	dlNetwork mnist("mnist");
	
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

	mnist.LoadInfo();

	float* inputData = new float[row * col];
	for (uint i = 0; i < times; i++)
	{
		for (uint j = 0; j < row * col; j++)
		{
			inputData[j] = (float)(*(inputImg + i * row * col + j));
		}
		Map<MatrixXf> inputMap(inputData, row * col, 1);
		MatrixXf inputM = inputMap;
		MatrixXf targatM = MatrixXf::Ones(10, 1);
		targatM = targatM * 0.1;
		targatM(input[i], 0) = 0.9;
		mnist.Test(inputM, targatM);
	}
	
	DBG_PRINT("%f\n", mnist.EpochStatistics());
	delete[] input;
	delete[] inputImg;
}

void main(int argc, char * argv[])
{
	//if (argc == 2)
	//{
	//	string str = argv[1];
	//	if (str == "t")
	//	{
	//		train();
	//	}
	//	else
	//		DBG_PRINT("Wrong argument!\n");
	//}
	//else if (argc == 1)
	//{
	//	test();
	//}
	//else
	//	DBG_PRINT("Wrong argument!\n");	
	Matrix3i m1;
	m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	Matrix2i m2;
	m2 << 1, 2, 3, 4;
	Conv<MatrixXd> re(a, b);

}