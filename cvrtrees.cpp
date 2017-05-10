#include "cvrtrees.h"

using namespace cv;


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
void trainRTrees(const Mat & trainSamples,
                const Mat & trainClasses,
                const CvRTParams & params,
                CvRTrees & rtrees)
{
    /* =================================================================== */
    //  �������� ���, ����������� �������� �������� ���������� ����.
    //  ��� �������� ������ �������������� ��� ���������� ��������� �������
    //  � ��� ��������, ����������� �������� ���, ��� �������� ����� ��������
    //  ��������������.
    /* ------------------------------------------------------------------- */






    /* =================================================================== */
}


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
int getRTreesPrediction(const Mat & sample,
                        const cv::ml::StatModel & model)
{
    // �������� ��� ������ � ���������� ����
    const CvRTrees & rf = dynamic_cast<const CvRTrees &>(model);
    int prediction = 0;
    /* =================================================================== */
    //  �������� ���, ����������� �������� ������������ ��� ���������� ����.
    /* ------------------------------------------------------------------- */






    /* =================================================================== */
    return prediction;
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
CvRTParams readRTreesParams()
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
    
    CvRTParams params(treeDepth, 0, 0.0f, false, 10, 0, false, 0, treesNum, 0.0f, CV_TERMCRIT_ITER);

    return params;
}
