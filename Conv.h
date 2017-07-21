#pragma once
#include "Eigen\core"
using namespace Eigen;

// cross-correlation
template<typename T>
T Conv(const T& matrixA, const T& matrixB)
{
	T matrixR;
	size_t r_row = matrixA.rows() - matrixB.rows() + 1;
	size_t r_col = matrixA.cols() - matrixB.cols() + 1;
	matrixR.resize(r_row, r_col);
	for (int i = 0; i < r_row; i++)
	{
		for (int j = 0; j < r_col; j++)
		{
			matrixR(i, j) = 0;
			for (int k1 = 0; k1 < matrixB.rows(); k1++)
			{
				for (int k2 = 0; k2 < matrixB.cols(); k2++)
					matrixR(i, j) += matrixB(k1, k2) * matrixA(i + k1, j + k2);
			}
		}
	}
	return matrixR;
}

template<typename T>
T Padding(const T& matrixA, int u, int v)
{
	size_t r_row = matrixA.rows() + 2 * u;
	size_t r_col = matrixA.cols() + 2 * v;
	T matrixR = T::Zero(r_row, r_col);
	matrixR.block(u, v, matrixA.rows(), matrixA.cols()) = matrixA;
	return matrixR;
}

// flip vertical
template<typename T>
T FlipV(const T& matrixA)
{
	T matrixR = matrixA;
	size_t r_row = matrixA.rows();
	size_t r_col = matrixA.cols();
	for (size_t i = 0; i < r_row; i++)
	{
		matrixR.block(r_row - 1 - i, 0, 1, r_col) = matrixA.block(i, 0, 1, r_col);
	}
	return matrixR;
}

template<typename T>
T EleWiseMul(const T& matrixA, const T& matrixB)
{
	T matrixR;	
	size_t r_row = matrixA.rows();
	size_t r_col = matrixA.cols();
	assert(r_row == matrixB.rows());
	assert(r_col == matrixB.cols());
	matrixR.resize(r_row, r_col);
	for (size_t i = 0; i < r_row; i++)
	{
		for (size_t j = 0; j < r_col; j++)
		{
			matrixR(i, j) = matrixA(i, j) * matrixB(i, j);
		}
	}
	return matrixR;
}