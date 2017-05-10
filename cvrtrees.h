#ifndef RANDOM_FOREST_TREE
#define RANDOM_FOREST_TREE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"


/*
// ������� ������ ���������� ��������� �������� ���������� ���� � �������.
// 
// API
// CvRTParams readRTreesParams()
// 
// ���������
// ��������� ��������� �������� ���������� ����.
// 
*/
CvRTParams readRTreesParams();


/*
// ������� �������� ���������� ����.
// 
// API
// void trainRTrees(const cv::Mat & trainSamples,
//                  const cv::Mat & trainClasses,
//                  const CvRTParams & params,
//                  CvRTrees & rtrees)
// 
// ����
// trainSamples  - ������� ������� ��������� ��� �����������
//                 ��������� �������
// trainClasses  - ������ ����������� ��������� �������
// params        - ��������� ��������� ��������
// 
// ����
// rtrees        - ��������� ��������� ���
*/
void trainRTrees(const cv::Mat & trainSamples,
                const cv::Mat & trainClasses,
                const CvRTParams & params,
                CvRTrees & rtrees);


/*
// ������� ������������ � ������� ���������� ����.
// 
// API
// int getRTreesPrediction(const cv::Mat & sample,
//                         const cv::ml::StatModel & model)
// 
// ����
// sample  - �������, ���������� ���������� ����� �����
//           � ������������ ���������
// model   - ��������� ��������� ���
// 
// ���������
// ������������� �����
*/
int getRTreesPrediction(const cv::Mat & sample,
                        const cv::ml::StatModel & model);


#endif