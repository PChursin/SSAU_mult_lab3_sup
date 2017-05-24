#include "cvgbtrees.h"

using namespace cv;


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
	Ptr<ml::Boost> boost)
                //const CvGBTreesParams & params,
                //CvGBTrees & gbtrees)
{
    /* =================================================================== */
    //  �������� ���, ����������� �������� �������� ������������ ��������
    //  �������� �������.
    //  ��� �������� ������ �������������� ��� ���������� ��������� �������
    //  � ��� ��������, ����������� �������� ���, ��� �������� ����� ��������
    //  ��������������.
    /* ------------------------------------------------------------------- */
	Ptr<ml::TrainData> trainData = ml::TrainData::create(trainSamples, ml::ROW_SAMPLE, trainClasses);
	boost->train(trainData);
    /* =================================================================== */
}

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
Ptr<ml::Boost> readGBTreesParams()
{
    int treeDepth = -1;
    printf("max depth of each tree = ");
    scanf("%d", &(treeDepth));

    int treesNum = -1;
    printf("number of trees to build = ");
    scanf("%d", &(treesNum));
	/*
    float learningRate = 0.0f;
    printf("learning rate (shrinkage) = ");
    scanf("%f", &(learningRate));

    float subsamplePortion = 0.0f;
    printf("subsample portion = ");
    scanf("%f", &(subsamplePortion));*/
    
	Ptr<ml::Boost> boost = ml::Boost::create();
	boost->setMaxDepth(treeDepth);
	boost->setWeakCount(treesNum);
	//boost->setBoostType(ml::Boost::Types::REAL);
	boost->setUseSurrogates(false);
	//boost->set
	/*
    CvGBTreesParams params(CvGBTrees::DEVIANCE_LOSS,
        treesNum,
        learningRate,
        subsamplePortion,
        treeDepth,
        false);*/

    return boost;
}
