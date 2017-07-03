#pragma once
#include "Eigen\core"
using namespace Eigen;

template<typename T>
Matrix<Scalar, Dynamic, Dynamic> Conv(const T& matrixA, const T& matrixB)
{
	Matrix<Scalar, Dynamic, Dynamic> matrixR;
	size_t r_row = matrixA.rows() - matrixB.rows() + 1;
	size_t r_col = matrixA.cols() - matrixB.cols() + 1;
	matrixR.resize(r_row, r_col);
	for (int i = 0; i < r_row; i++)
	{
		for (int j = 0; j < r_col; j++)
		{
			r(i, j) = 0;
			for (int k1 = 0; k1 < matrixB.rows(); k1++)
			{
				for (int k2 = 0; k2 < matrixB.cols(); k2++)
					r(i, j) += b(k1, k2) * a(i - k1 + 1, j - k2 + 1);
			}
		}
	}
}
	//template<typename T> class Conv
	//{
	//public:
	//	Conv(const T& matrixA, const T& matrixB)
	//	{
	//		A_Mat = &matrixA;
	//		B_Mat = &matrixB;
	//		R_Mat = new Matrix<Scalar, Dynamic, Dynamic>();
	//	}
	//	~Conv()
	//	{
	//		if (R_Mat)
	//			delete R_Mat;
	//	}
	//	const Matrix<Scalar, Dynamic, Dynamic>& Eval() 
	//	{
	//		return *R_Mat;
	//	}

	//private:
	//	void TwoMatrixConv()
	//	{
	//		typedef typename T::Scalar Scalar;
	//		Matrix<Scalar, Dynamic, Dynamic>& r = *R_Mat;
	//		const T& a = *A_Mat;
	//		const T& b = *B_Mat;
	//		r.resize(R_Row, R_Col);
	//		for (int i = 0; i < R_Row; i++)
	//		{
	//			for (int j = 0; j < R_Col; j++)
	//			{
	//				r(i, j) = 0;
	//				for (int k1 = 0; k1 < b.rows(); k1++)
	//				{
	//					for (int k2 = 0; k2 < b.cols(); k2++)
	//						r(i, j) += b(k1, k2) * a(i - k1 + 1, j - k2 + 1);
	//				}
	//			}
	//		}
	//	}
	//	const T* A_Mat;
	//	const T* B_Mat;
	//	Matrix<Scalar, Dynamic, Dynamic>* R_Mat;
	//};
