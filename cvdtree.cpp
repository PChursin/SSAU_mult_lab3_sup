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
// ������� ������������ � ������� ������ �������.
// 
// API
// int getDTreePrediction(const cv::Mat & sample,
//                        const cv::ml::StatModel & model)
// 
// ����
// sample  - �������, ���������� ���������� ����� �����
//           � ������������ ���������
// model   - ��������� ������ �������
// 
// ���������
// ������������� �����
*/
int getDTreePrediction(const Mat & sample, const cv::ml::StatModel & model)
{
    // �������� ��� ����. ������ � ������ �������
    //const CvDTree & dtree = dynamic_cast<const CvDTree &>(model);
    int prediction = 0;
    /* =================================================================== */
    //  �������� ���, ����������� �������� ������������ ��� ������ �������.
    /* ------------------------------------------------------------------- */

	Mat out;
	model.predict(sample, out);
	prediction = out.at<int>(0);




    /* =================================================================== */
    return prediction;
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
    params->setTruncatePrunedTree(true);
    // �������� ��������� ��������� ������.
    params->setUse1SERule(true);

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
    int doPruning = 0;
    printf("apply pruning (0/1) = ");
    scanf("%d", &(doPruning));
    // �.�. ������ � �������� ������� �������,
    // ���������� ����������� ���������� ��������.
    params->setCVFolds((doPruning == 0) ? 0 : 5);

    return params;
}

