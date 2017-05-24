#ifndef GRADIENT_BOOSTED_TREES
#define GRADIENT_BOOSTED_TREES

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;


/*
// ������� ������ ���������� ��������� �������� ������������
// �������� �������� ������� � �������.
// 
// API
// CvGBTreesParams readGBTreesParams()
// 
// ���������
// ��������� ��������� �������� ������������ ��������
// �������� �������.
// 
*/
Ptr<ml::Boost> readGBTreesParams();


/*
// ������� �������� ������������ ��������
// �������� �������.
// 
// API
// void trainGBTrees(const cv::Mat & trainSamples,
//                  const cv::Mat & trainClasses,
//                  const CvGBTreesParams & params,
//                  CvGBTrees & gbtrees)
// 
// ����
// trainSamples  - ������� ������� ��������� ��� �����������
//                 ��������� �������
// trainClasses  - ������ ����������� ��������� �������
// params        - ��������� ��������� ��������
// 
// ����
// gbtrees        - ��������� ������ ������������ ��������
*/
void trainGBTrees(const cv::Mat & trainSamples,
	const cv::Mat & trainClasses,
	Ptr<ml::Boost> boost);
                //const CvGBTreesParams & params,
                //CvGBTrees & gbtrees);



#endif