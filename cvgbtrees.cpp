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
                const CvGBTreesParams & params,
                CvGBTrees & gbtrees)
{
    /* =================================================================== */
    //  �������� ���, ����������� �������� �������� ������������ ��������
    //  �������� �������.
    //  ��� �������� ������ �������������� ��� ���������� ��������� �������
    //  � ��� ��������, ����������� �������� ���, ��� �������� ����� ��������
    //  ��������������.
    /* ------------------------------------------------------------------- */







    /* =================================================================== */
}


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
                        const cv::ml::StatModel & model)
{
    // �������� ��� ������ � ���������� ����
    const CvGBTrees & gbtrees = dynamic_cast<const CvGBTrees &>(model);
    int prediction = 0;
    /* =================================================================== */
    //  �������� ���, ����������� �������� ������������ ��� ���������� ����.
    /* ------------------------------------------------------------------- */





    /* =================================================================== */
    return prediction;
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
CvGBTreesParams readGBTreesParams()
{
    int treeDepth = -1;
    printf("max depth of each tree = ");
    scanf("%d", &(treeDepth));

    int treesNum = -1;
    printf("number of trees to build = ");
    scanf("%d", &(treesNum));

    float learningRate = 0.0f;
    printf("learning rate (shrinkage) = ");
    scanf("%f", &(learningRate));

    float subsamplePortion = 0.0f;
    printf("subsample portion = ");
    scanf("%f", &(subsamplePortion));
    
    CvGBTreesParams params(CvGBTrees::DEVIANCE_LOSS,
        treesNum,
        learningRate,
        subsamplePortion,
        treeDepth,
        false);

    return params;
}
