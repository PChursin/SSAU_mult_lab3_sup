#include "cvrtrees.h"
using namespace cv;
using namespace cv::ml;


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
	Ptr<cv::ml::RTrees> rtrees)
{
    /* =================================================================== */
    //  �������� ���, ����������� �������� �������� ���������� ����.
    //  ��� �������� ������ �������������� ��� ���������� ��������� �������
    //  � ��� ��������, ����������� �������� ���, ��� �������� ����� ��������
    //  ��������������.
    /* ------------------------------------------------------------------- */

	Ptr<ml::TrainData> trainData = ml::TrainData::create(trainSamples, ml::ROW_SAMPLE, trainClasses);
	rtrees->train(trainData);



    /* =================================================================== */
}


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
Ptr<RTrees> readRTreesParams()
{
    int treeDepth = -1;
    printf("max depth of each tree = ");
    scanf("%d", &(treeDepth));

    int treesNum = -1;
    printf("number of trees to build = ");
    scanf("%d", &(treesNum));

    int activeVarsNum = -1;
    printf("number of active variables (set 0 to use the sqrt(total number of features)) = ");
    scanf("%d", &(activeVarsNum));
    
	Ptr<RTrees> params = RTrees::create();
	params->setMaxDepth(treeDepth);
	params->setActiveVarCount(activeVarsNum);
	params->setMinSampleCount(0);

    //CvRTParams params(treeDepth, 0, 0.0f, false, 10, 0, false, 0, treesNum, 0.0f, CV_TERMCRIT_ITER);

    return params;
}
