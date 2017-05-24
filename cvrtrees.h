#ifndef RANDOM_FOREST_TREE
#define RANDOM_FOREST_TREE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;
using namespace cv::ml;

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
Ptr<RTrees> readRTreesParams();


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
				Ptr<RTrees> rtrees);
                //const CvRTParams & params,
                //CvRTrees & rtrees);



#endif