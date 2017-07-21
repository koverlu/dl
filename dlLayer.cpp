#include "dlLayer.h"
#include "debug.h"
#include <iostream>
#include "Conv.h"
double activ_sigmoid(double x)
{
	return (1 / (1 + exp(-x)));
}

double activ_relu(double x)
{
	return x >= 0 ? x : 0;
}

uint dlLayer::m_sLayerCnt = 0;

dlLayer::dlLayer(dlLayerType type, Vector3i inDim, Vector3i filterDim, ActivatorType activator, dlLayer* pUpLayer) :
	m_type(type),
	m_inDim(inDim),
	m_filterDim(filterDim),
	m_pUpLayer(pUpLayer),
	m_rate(0.1),
	m_outDim(Vector3i(0,0,0)),
	m_layerId(m_sLayerCnt++)
{
	if (pUpLayer)
	{
		if (pUpLayer->m_pDownLayer == NULL)
			pUpLayer->m_pDownLayer = this;
		else
			DBG_ASSERT(pUpLayer->m_pDownLayer == this, "UpLayer set error!");
	}
	if (activator == ACTIV_SIGMOID)
	{
		m_Activator = activ_sigmoid;
	}
	else if (activator == ACTIV_RELU)
	{
		m_Activator = activ_relu;
	}
}

void dlLayer::Save()
{
}

void dlLayer::GradientCheck()
{
	const double epsilon = 0.0001;
	dlInputLayer inputLayer(m_inDim);
	m_pUpLayer = &inputLayer;
	for (size_t i = 0; i < m_inDim.z(); i++)
	{
		MatrixXf inputMat = MatrixXf::Random(m_inDim.x(), m_inDim.y()) * 2;
		std::cout << inputMat << std::endl << "====================\n";
		inputLayer.SetInputData(i, inputMat);
	}
	Forward();
	vector<dlFilter> Filter_deriv;
	vector<MatrixXf> output;
	for (size_t i = 0; i < m_outDim.z(); i++)
	{
		std::cout << m_vData[i] << std::endl << "====================\n";
		MatrixXf outputMat = MatrixXf::Random(m_outDim.x(), m_outDim.y()) * 2;
		MatrixXf Deviation = outputMat - m_vData[i];
		dlFilter flt;		
		for (size_t k = 0; k < m_inDim.z(); k++)
		{
			MatrixXf weight = Conv(m_pUpLayer->m_vData[k], Deviation);
			flt.weights.push_back(weight);
		}
		Filter_deriv.push_back(flt);
		output.push_back(outputMat);
	}

	for (size_t i = 0; i < m_outDim.z(); i++)
	{
		for (size_t m = 0; m < m_filterDim.z(); m++)
		{
			for (size_t j = 0; j < m_filterDim.x(); j++)
			{
				for (size_t k = 0; k < m_filterDim.y(); k++)
				{
					m_vFilter[i].weights[m](j, k) += epsilon;
					Forward();
					MatrixXf d = output[i] - m_vData[i];
					double k2 = d.sum();
					double err1 = (output[i] - m_vData[i]).sum();
					m_vFilter[i].weights[m](j, k) -= 2 * epsilon;
					Forward();
					double err2 = (output[i] - m_vData[i]).sum();
					double expect = (err1 - err2) / (2 * epsilon);
					DBG_PRINT("filter %d:weight(%d, %d, %d): exp - act %f - %f\n", i, j, k, m, expect, Filter_deriv[i].weights[m](j, k));
				}
			}
		}

	}

}

void dlLayer::SetFilter(uint idx, dlFilter & flt)
{
	DBG_ASSERT(idx <= m_outDim.z() && flt.weights.size() == m_filterDim.z(), "Filter size is invalid!\n");
	m_vFilter[idx].weights = flt.weights;
	m_vFilter[idx].bias = flt.bias;
}

dlInputLayer::dlInputLayer(Vector3i outDim):
	dlLayer(DL_INPUT, Vector3i(0, 0, 0), Vector3i(0, 0, 0), ACTIV_NONE, NULL)
{
	m_outDim = outDim;
	for (uint i = 0; i < outDim.z(); i++)
	{
		m_vData.push_back(MatrixXf::Zero(outDim.x(), outDim.y()));
	}
}

void dlInputLayer::SetInputData(uint idx, const MatrixXf & mat)
{
	DBG_ASSERT(idx <= m_outDim.z() && mat.rows() == m_outDim.x() && mat.cols() == m_outDim.y(), "Invalid input data!\n");
	m_vData[idx] = mat;
}
