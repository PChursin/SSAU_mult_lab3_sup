#ifndef DECISION_TREE
#define DECISION_TREE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;

/*
// ������� ������ ���������� ��������� �������� ������ ������� � �������.
// 
// API
// CvSVMParams readSVMParams()
// 
// ���������
// ��������� ��������� �������� ������ �������.
// 
*/
Ptr<ml::DTrees> readDTreeParams();


/*
// ������� �������� ������ �������.
// 
// API
// void trainDTree(const cv::Mat & trainSamples,
//                 const cv::Mat & trainClasses,
//                 const CvDTreeParams & params,
//                 CvDTree & dtree)
// 
// ����
// trainSamples  - ������� ������� ��������� ��� �����������
//                 ��������� �������
// trainClasses  - ������ ����������� ��������� �������
// params        - ��������� ��������� ��������
// 
// ����
// dtree         - ��������� ������ �������
*/
void trainDTree(const cv::Mat & trainSamples,
	const cv::Mat & trainClasses,
	const cv::Ptr<ml::DTrees> params);
                //CvDTree & dtree);

#endif