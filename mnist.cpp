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
	
	const uint batchSize = 128;
	dlNetwork mnist("mnist", batchSize);
	//Create Network, LSTM + FC
	mnist.Init();
	//Create Input Buffer and Target Buffer
	vector<double> inputData(row * col * batchSize);
	vector<double> targetData(10 * batchSize);
	mnist.SetInputsAndTargets(&inputData, &targetData);

	uint epoch = 0;
	double errorRate = 0;
	double lastErrorRate = 1.0f;
	vector<uint> rand_idx(times);
	for (uint i = 0; i < times; i++)
		rand_idx[i] = i;	
	uint total_start = GetTickCount();
	while (1)
	{
		uint epoch_start = GetTickCount();
		random_shuffle(rand_idx.begin(), rand_idx.end());
		//! Manully check if the image match the lable
		//for (uint i = 0; i < times / 128; i++)
		//{
		//	for (uint k = 0; k < 128; k++)
		//	{
		//		for (uint m = 0; m < row; m++)
		//		{
		//			for (uint n = 0; n < col; n++)
		//			{
		//				if ((float)(*(inputImg + rand_idx[i * 128 + k] * row * col + m * col + n)) > 0)
		//					cout << 1 << " ";
		//				else
		//					cout << " " << " ";
		//			}
		//			cout << endl;
		//		}
		//		cout << ">>>>>>>>>> " << (int)input[rand_idx[i * 128 + k]] << " <<<<<<<<<<" << endl;
		//	}
		//}

		for (uint i = 0; i < times / batchSize; i++)
		{
			memset(&targetData[0], 0, 10 * batchSize * sizeof(double));
			for (uint k = 0; k < batchSize; k++)
			{
				for (uint j = 0; j < row * col; j++)
				{
					inputData[k * row * col + j] = (float)(*(inputImg + rand_idx[i * batchSize + k] * row * col + j));
				}
				targetData[input[rand_idx[i * batchSize + k]] + k * 10] = 1;
			}			
			mnist.Train();
		}
		if(mnist.EpochStatistics() < 0.1)
			break;

		//if (epoch % 10 == 0)
		//{
		//	errorRate = 0;
		//	errorRate = mnist.GetLastErrorRate(5);
		//	if (errorRate > lastErrorRate)
		//		break;
		//	else
		//		lastErrorRate = errorRate;
		//}
		//if (mnist.GetLastErrorRate(5) < 0.1)
		//{
			//break;
		//}
		epoch++;
		mnist.SaveInfo();
		uint epoch_end = GetTickCount();
		DBG_PRINT("EPOCH %d: %d Seconds\n", epoch, (epoch_end - epoch_start) / 1000);
	}
	uint total_end = GetTickCount();
	DBG_PRINT("Total cpu time: %d Seconds\n", (total_end - total_start) / 1000);
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

int main(int argc, char * argv[])
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
	return 1;
}