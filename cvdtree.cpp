#include "cvdtree.h"

using namespace cv;


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
void trainDTree(const Mat & trainSamples,
                const Mat & trainClasses,
                //const CvDTreeParams & params,
                //CvDTree & dtree)
				Ptr<ml::DTrees> dtree)
{
    /* =================================================================== */
    //  �������� ���, ����������� �������� �������� ������ �������.
    //  ��� �������� ������ �������������� ��� ���������� ��������� �������
    //  � ��� ��������, ����������� �������� ���, ��� �������� ����� ��������
    //  ��������������.
    /* ------------------------------------------------------------------- */
	Ptr<ml::TrainData> trainData = ml::TrainData::create(trainSamples, ml::ROW_SAMPLE, trainClasses);
	dtree->train(trainData);

    /* =================================================================== */
}


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
Ptr<ml::DTrees> readDTreeParams()
{
	Ptr<ml::DTrees> params = ml::DTrees::create();
	
    //CvDTreeParams params;
    
    // �� ������ ����������� ���������.
    params->setUseSurrogates(false);
    // �� ������ ����������, ��������� � ���� ������� ������ (pruning).
    params->setTruncatePrunedTree(false);
    // �������� ��������� ��������� ������.
    params->setUse1SERule(false);
	
    // ������������ ������ (�������) ������.
	int tmp;
    printf("maximal tree depth = ");
    scanf("%d", &tmp);
	params->setMaxDepth(tmp);
	
    // ����������� ���������� ����������� ��������� �������,
    // ��� ������� ����������� ���������� ���������.
    printf("minimal number of samples in leaf = ");
    scanf("%d", &tmp);
	params->setMinSampleCount(tmp);


    // ���������� ������ � ���������� �������� (cross-validation).
    //int doPruning = 0;
    //printf("apply pruning (0/1) = ");
    //scanf("%d", &(doPruning));
    // �.�. ������ � �������� ������� �������,
    // ���������� ����������� ���������� ��������.
    //params->setCVFolds((doPruning == 0) ? 0 : 5);
	params->setCVFolds(0);

    return params;
}

