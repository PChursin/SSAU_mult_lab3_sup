#include <stdio.h>
#include <map>
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cvsvm.h"
#include "cvdtree.h"
//#include "cvrtrees.h"
//#include "cvgbtrees.h"
#include "auxilary.h"
#include "errorMetrics.h"

#pragma warning(disable : 4996)

using namespace cv;
using std::map;


// ������������ ����� ������ � ��������� ������������ ����
const int maxWinNameLen = 1000;
// ��� ������� ESC
const int escCode = 27;

// ������������ �������� �������� ��������� � ������� ����
enum {LOAD_DATA = 0, USE_SVM, USE_DTREE, USE_RTREES, USE_GBTREES};


/*
// ������� ������ �������� ���� � ������ ������ �������������.
// 
// API
// int getMainMenuAction()
//
// ���������
// ����� ���������� ������������� ������ ����.
*/
int getMainMenuAction()
{
    int menuItemIdx = -1;
    const int menuTabsNum = 5;
    const char* menu[] = 
    {
        "0 - Load data", 
        "1 - Use SVM", 
        "2 - Use decision tree",
        "3 - Use random forest",
        "4 - Use gradient boosted trees"
    };

    while (true)
    {
        // �������� �������� ����
        printf("Main menu:\n");
        for (int i = 0; i < menuTabsNum; i++)
        {
            printf("\t%s\n", menu[i]);
        }
        printf("\n");
        // ����� ������������
        printf("Your choice = ");
        scanf("%d", &menuItemIdx);
        // �������� �� ����� ����������� ������
        if (menuItemIdx >= 0 && menuItemIdx < menuTabsNum)
        {
            break;
        }
    }
    return menuItemIdx;
}


int main(int argc, char** argv)
{
    // ������� ��� �������� ��������� � �������� �������
    Mat featuresTrain;
    Mat classesTrain;
    Mat featuresTest;
    Mat classesTest;

    // ���-�� ������� ������� ���� ����������� �� ������� ������
    int svmModelsNum = 0;
    int dtreeModelsNum = 0;
    int rtreesModelsNum = 0;
    int gbtreesModelsNum = 0;

    // ��� �������� ������������ ����
    char winName[maxWinNameLen] = {0};

    // ����������� ������� ������� � BGR-�����
    map<int, Scalar> classColors;
    classColors[0] = Scalar(255, 191, 0);
    classColors[1] = Scalar(0, 215, 255);
    classColors[2] = Scalar(71, 99, 255);
    classColors[3] = Scalar(0, 252, 124);
    classColors[4] = Scalar(240, 32, 160);

    // ��������� �� ������
    bool isDataLoaded = false;

    int ans;
    do
    {
        // ����� ������� ������ ������ ����
        int activeMenuItem = getMainMenuAction();
        Ptr<ml::StatModel> model;
        //getPredictedClassLabel * predictFunction;
		PredictionFunction * predictFunction;

        switch (activeMenuItem)
        {
        case LOAD_DATA:
            {
                // ��������� ��� ����������� ����.
                destroyAllWindows();
                // ������ ��������� � �������� ������� �� �����.
                readDatasetFromFile(featuresTrain,
                    classesTrain,
                    featuresTest,
                    classesTest);
                isDataLoaded = true;
                // �������� �������� ���������� �������,
                // ��������� �� ����������� ������.
                svmModelsNum = 0;
                dtreeModelsNum = 0;
                rtreesModelsNum = 0;
                gbtreesModelsNum = 0;
                continue;
            }; break;

        case USE_SVM:
            {
                // ����� ������������ �������� ���������� ��������� ������.
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }
                // ����������� ��������� ��������� ��������.
                // CvSVMParams params = readSVMParams();
				Ptr<ml::SVM> svm = readSVMParams();
                // ��������� �������� ��������.
                //CvSVM * svm = new CvSVM;
                //trainSVM(featuresTrain, classesTrain, params, *svm);
				Ptr<ml::TrainData> trainData = ml::TrainData::create(featuresTrain, ml::ROW_SAMPLE, classesTrain);
				svm->trainAuto(trainData);
				//trainSVM(featuresTrain, classesTrain, svm);
                sprintf(winName, "SVM #%d\0", svmModelsNum++);
                model = svm;
				predictFunction = new SVMPrediction();
                //predictFunction = getSVMPrediction;
            }; break;
			
        case USE_DTREE:
            {
                // ����� ������������ �������� ���������� ��������� ������
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }

                // ����������� ��������� ��������� ��������
                //CvDTreeParams params = readDTreeParams();

                // ��������� �������� ��������
				Ptr<ml::DTrees> dtree = readDTreeParams();
                //CvDTree * dtree = new CvDTree;
                trainDTree(featuresTrain,
                    classesTrain,
	                dtree);

                sprintf(winName, "Decision tree #%d\0", dtreeModelsNum++);
                model = dtree;
                predictFunction = new DTreePrediction();
            }; break;
			/*
        case USE_RTREES:
            {
                // ����� ������������ �������� ���������� ��������� ������
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }

                // ����������� ��������� ��������� ��������
                CvRTParams params = readRTreesParams();

                // ��������� �������� ��������
                CvRTrees * rtrees = new CvRTrees;
                trainRTrees(featuresTrain, classesTrain, params, *rtrees);

                sprintf(winName, "Random forest #%d\0", rtreesModelsNum++);
                model = rtrees;
                predictFunction = getRTreesPrediction;
            }; break;
        case USE_GBTREES:
            {
                // ����� ������������ �������� ���������� ��������� ������
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }

                // ����������� ��������� ��������� ��������
                CvGBTreesParams params = readGBTreesParams();

                // ��������� �������� ��������
                CvGBTrees * gbtrees = new CvGBTrees;
                trainGBTrees(featuresTrain, classesTrain, params, *gbtrees);

                sprintf(winName, "Gradient boosting #%d\0", gbtreesModelsNum++);
                model = gbtrees;
                predictFunction = getGBTreesPrediction;
            }; break;*/
        }

        // ��������� ������ �� ��������� � �������� ��������.
        float trainError = getClassificationError(featuresTrain,
            classesTrain,
            model,
			false,
            predictFunction);
        float testError = getClassificationError(featuresTest,
            classesTest,
            model,
			true,
            predictFunction);
        
        printf("========== %s ===========\n", winName);
        printf("error on train set: %.3f\n", trainError);
        printf("error on test set: %.3f\n", testError);

        // ���� ������������ ��������� ��������� -- ������ ������������
        if (featuresTrain.cols == 2)
        {
            // ������� ����� ����������� � ����������� 400�400
            Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));
            // ������ ��������� ����������� ��������� �� �������
            drawPartition(img,
                classColors,
                getRanges(featuresTest),
                Size(77, 77),
                model,
                predictFunction);
            // ������������ ����� �������� �������
            drawPoints(img, featuresTest, classesTest, getRanges(featuresTest), classColors, 1);
            // ������������ ����� ��������� �������
            drawPoints(img, featuresTrain, classesTrain, getRanges(featuresTest), classColors, 0);

            //CvSVM * svm = dynamic_cast<CvSVM*>(model);
			Ptr<ml::SVM> svm = model.dynamicCast<ml::SVM>();
            if (svm)
            {
                // ������������ ������� �������
                Mat supportVectors = getSupportVectors(svm);
                drawPoints(img, supportVectors, classesTrain, getRanges(featuresTest), classColors, 2);
                //svm = 0;
            }
            //*/
            // ������� ����������� � ����� ����
            namedWindow(winName);
            imshow(winName, img);
        }
        //delete model;
        //model = 0;
        predictFunction = 0;

        printf("Do you want to continue? ESC - exit\n");
        // ������� ������� ������� � ����������� ����.
        ans = waitKey();
    }
    while (ans != escCode);

    // ��������� ��� ����������� ����.
    destroyAllWindows();
    return 0;
}