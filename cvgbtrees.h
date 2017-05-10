#ifndef GRADIENT_BOOSTED_TREES
#define GRADIENT_BOOSTED_TREES

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"


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
CvGBTreesParams readGBTreesParams();


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
                const CvGBTreesParams & params,
                CvGBTrees & gbtrees);


/*
// ������� ������������ � ������� ������ ������������ ��������
// �������� �������.
// 
// API
// int getGBTreesPrediction(const cv::Mat & sample,
//                         const cv::ml::StatModel & model)
// 
// ����
// sample  - �������, ���������� ���������� ����� �����
//           � ������������ ���������
// model   - ��������� ������ ������������ ��������
// 
// ���������
// ������������� �����
*/
int getGBTreesPrediction(const cv::Mat & sample,
                        const cv::ml::StatModel & model);


#endif