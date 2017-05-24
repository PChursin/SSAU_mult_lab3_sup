#include <stdio.h>
#include <map>
#include <fstream>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cvsvm.h"
#include "cvdtree.h"
#include "cvrtrees.h"
#include "cvgbtrees.h"
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
enum {LOAD_DATA = 0, USE_SVM, USE_DTREE, USE_RTREES, USE_GBTREES, LOAD_MODEL, SAVE_MODEL};


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
    const int menuTabsNum = 7;
    const char* menu[] = 
    {
        "0 - Load data", 
        "1 - Use SVM", 
        "2 - Use decision tree",
        "3 - Use random forest",
        "4 - Use gradient boosted trees",
		"5 - Load model",
		"6 - Save model"
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
        printf("Your choice = ");
        scanf("%d", &menuItemIdx);
        // �������� �� ���� ����������� ������
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
	Ptr<ml::StatModel> model;
    do
    {
        // ����� ������� ������ ������ ����
        int activeMenuItem = getMainMenuAction();

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
				model.release();
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
				printf("Enter 0 to use auto train params, 1 - manual: ");
				int manual;
				scanf("%d", &manual);
				Ptr<ml::SVM> svm = readSVMParams(!manual);
				if (!manual) {
					// ��������� �������� ��������.
					Ptr<ml::TrainData> trainData = ml::TrainData::create(featuresTrain, ml::ROW_SAMPLE, classesTrain);
					svm->trainAuto(trainData);
				}
				else {
					trainSVM(featuresTrain, classesTrain, svm);
				}

                sprintf(winName, "SVM #%d\0", svmModelsNum++);
                model = svm;
            }; break;
			
        case USE_DTREE:
            {
                // ����� ������������ �������� ���������� ��������� ������
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }

				Ptr<ml::DTrees> dtree = readDTreeParams();

                trainDTree(featuresTrain,
                    classesTrain,
	                dtree);

                sprintf(winName, "Decision tree #%d\0", dtreeModelsNum++);
                model = dtree;
            }; break;
			
        case USE_RTREES:
            {
                // ����� ������������ �������� ���������� ��������� ������
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }

                // ����������� ��������� ��������� ��������
                Ptr<ml::RTrees> rtrees = readRTreesParams();
				// ��������� �������� ��������
				trainRTrees(featuresTrain, classesTrain, rtrees);

                sprintf(winName, "Random forest #%d\0", rtreesModelsNum++);
                model = rtrees;
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
				Ptr<ml::Boost> boost = readGBTreesParams();
                // ��������� �������� ��������
                trainGBTrees(featuresTrain, classesTrain, boost);

                sprintf(winName, "Gradient boosting #%d\0", gbtreesModelsNum++);
                model = boost;
            }; break;
		case LOAD_MODEL:
			{
				if (!isDataLoaded)
				{
					printf("data must be loaded first\n");
					continue;
				}
				char filename[1000];
				printf("File = ");
				scanf("%s", &filename);
				FileStorage fs;
				fs.open(filename, FileStorage::READ);
				if (fs.isOpened() == false)
				{
					printf("No such file!\n");
					continue;
				}
				std::wfstream file(filename);
				wchar_t buf[100];
				file.getline(buf, 100);
				file.getline(buf, 100);
				file.getline(buf, 100);
				std::wcout << buf << std::endl;
				std::wstring readHead = buf;
				std::wstring svmHead = L"opencv_ml_svm:";
				std::wstring dtreeHead = L"opencv_ml_dtree:";
				std::wstring rtreeHead = L"opencv_ml_rtree:";
				std::wstring boostHead = L"opencv_ml_boost:";
				if (readHead == svmHead) {
					model = ml::StatModel::load<ml::SVM>(filename);
					sprintf(winName, "SVM #%d\0", svmModelsNum++);
				}
				else if (readHead == dtreeHead) {
					model = ml::StatModel::load<ml::DTrees>(filename);
					sprintf(winName, "Decision tree #%d\0", dtreeModelsNum++);
				}
				else if (readHead == rtreeHead) {
					model = ml::StatModel::load<ml::RTrees>(filename);
					sprintf(winName, "Random forest #%d\0", rtreesModelsNum++);
				}
				else if (readHead == boostHead) {
					model = ml::StatModel::load<ml::Boost>(filename);
					sprintf(winName, "Gradient boosting #%d\0", gbtreesModelsNum++);
				}

			}break;
		case SAVE_MODEL:
		{
			if (model.get() == NULL) {
				printf("Train the model first\n");
				continue;
			}
			char filename[1000];
			printf("File = ");
			scanf("%s", &filename);
			model->save(filename);
			continue;
		}break;
        }

        // ��������� ������ �� ��������� � �������� ��������.
		float trainError = getClassificationError(featuresTrain,
			classesTrain,
			model,
			false);
		float testError = getClassificationError(featuresTest,
			classesTest,
			model,
			true);
        
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
				model);
            // ������������ ����� �������� �������
            drawPoints(img, featuresTest, classesTest, getRanges(featuresTest), classColors, 1);
            // ������������ ����� ��������� �������
            drawPoints(img, featuresTrain, classesTrain, getRanges(featuresTest), classColors, 0);

			Ptr<ml::SVM> svm = model.dynamicCast<ml::SVM>();
            if (svm)
            {
                // ������������ ������� �������
                Mat supportVectors = getSupportVectors(svm);
                drawPoints(img, supportVectors, classesTrain, getRanges(featuresTest), classColors, 2);
            }
            //*/
            // ������� ����������� � ����� ����
            namedWindow(winName);
            imshow(winName, img);
        }

        printf("Do you want to continue? ESC - exit\n");
        // ������� ������� ������� � ����������� ����.
        ans = waitKey();
    }
    while (ans != escCode);

    // ��������� ��� ����������� ����.
    destroyAllWindows();
    return 0;
}