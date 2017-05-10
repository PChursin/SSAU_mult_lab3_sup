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


// Максимальная длина строки с названием графического окна
const int maxWinNameLen = 1000;
// Код символа ESC
const int escCode = 27;

// Перечисление задающее действие выбранное в главном меню
enum {LOAD_DATA = 0, USE_SVM, USE_DTREE, USE_RTREES, USE_GBTREES};


/*
// Функция печати главного меню и выбора пунтка пользователем.
// 
// API
// int getMainMenuAction()
//
// РЕЗУЛЬТАТ
// Номер выбранного пользователем пункта меню.
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
        // Печатаем основное меню
        printf("Main menu:\n");
        for (int i = 0; i < menuTabsNum; i++)
        {
            printf("\t%s\n", menu[i]);
        }
        printf("\n");
        // Выбор пользователя
        printf("Your choice = ");
        scanf("%d", &menuItemIdx);
        // Проверка на ввыод корректного номера
        if (menuItemIdx >= 0 && menuItemIdx < menuTabsNum)
        {
            break;
        }
    }
    return menuItemIdx;
}


int main(int argc, char** argv)
{
    // матрицы для хранения обучающей и тестовой выборок
    Mat featuresTrain;
    Mat classesTrain;
    Mat featuresTest;
    Mat classesTest;

    // кол-ва моделей каждого типа построенных на текущих данных
    int svmModelsNum = 0;
    int dtreeModelsNum = 0;
    int rtreesModelsNum = 0;
    int gbtreesModelsNum = 0;

    // имя текущего графического окна
    char winName[maxWinNameLen] = {0};

    // отображение номеров классов в BGR-цвета
    map<int, Scalar> classColors;
    classColors[0] = Scalar(255, 191, 0);
    classColors[1] = Scalar(0, 215, 255);
    classColors[2] = Scalar(71, 99, 255);
    classColors[3] = Scalar(0, 252, 124);
    classColors[4] = Scalar(240, 32, 160);

    // загружены ли данные
    bool isDataLoaded = false;

    int ans;
    do
    {
        // вызов функции выбора пункта меню
        int activeMenuItem = getMainMenuAction();
        Ptr<ml::StatModel> model;
        //getPredictedClassLabel * predictFunction;
		PredictionFunction * predictFunction;

        switch (activeMenuItem)
        {
        case LOAD_DATA:
            {
                // Закрываем все графические окна.
                destroyAllWindows();
                // Читаем обучающую и тестовую выборки из файла.
                readDatasetFromFile(featuresTrain,
                    classesTrain,
                    featuresTest,
                    classesTest);
                isDataLoaded = true;
                // Обнуляем счетчики количества моделей,
                // обученных на загруженных данных.
                svmModelsNum = 0;
                dtreeModelsNum = 0;
                rtreesModelsNum = 0;
                gbtreesModelsNum = 0;
                continue;
            }; break;

        case USE_SVM:
            {
                // Перед выполенением обучения необходимо загрузить данные.
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }
                // Запрашиваем параметры алгоритма обучения.
                // CvSVMParams params = readSVMParams();
				Ptr<ml::SVM> svm = readSVMParams();
                // Запускаем алгоритм обучения.
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
                // Перед выполенением обучения необходимо загрузить данные
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }

                // Запрашиваем параметры алгоритма обучения
                //CvDTreeParams params = readDTreeParams();

                // Запускаем алгоритм обучения
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
                // Перед выполенением обучения необходимо загрузить данные
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }

                // Запрашиваем параметры алгоритма обучения
                CvRTParams params = readRTreesParams();

                // Запускаем алгоритм обучения
                CvRTrees * rtrees = new CvRTrees;
                trainRTrees(featuresTrain, classesTrain, params, *rtrees);

                sprintf(winName, "Random forest #%d\0", rtreesModelsNum++);
                model = rtrees;
                predictFunction = getRTreesPrediction;
            }; break;
        case USE_GBTREES:
            {
                // Перед выполенением обучения необходимо загрузить данные
                if (!isDataLoaded)
                {
                    printf("data must be loaded first\n");
                    continue;
                }

                // Запрашиваем параметры алгоритма обучения
                CvGBTreesParams params = readGBTreesParams();

                // Запускаем алгоритм обучения
                CvGBTrees * gbtrees = new CvGBTrees;
                trainGBTrees(featuresTrain, classesTrain, params, *gbtrees);

                sprintf(winName, "Gradient boosting #%d\0", gbtreesModelsNum++);
                model = gbtrees;
                predictFunction = getGBTreesPrediction;
            }; break;*/
        }

        // Вычисляем ошибки на обучающей и тестовой выборках.
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

        // Если пространство признаков двумерное -- делаем визуализацию
        if (featuresTrain.cols == 2)
        {
            // Создаем белое изображение с разрешением 400х400
            Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));
            // Рисуем разбиение пространсва признаков на области
            drawPartition(img,
                classColors,
                getRanges(featuresTest),
                Size(77, 77),
                model,
                predictFunction);
            // Отрисовываем точки тестовой выборки
            drawPoints(img, featuresTest, classesTest, getRanges(featuresTest), classColors, 1);
            // Отрисовываем точки обучающей выборки
            drawPoints(img, featuresTrain, classesTrain, getRanges(featuresTest), classColors, 0);

            //CvSVM * svm = dynamic_cast<CvSVM*>(model);
			Ptr<ml::SVM> svm = model.dynamicCast<ml::SVM>();
            if (svm)
            {
                // Отрисовываем опорные векторы
                Mat supportVectors = getSupportVectors(svm);
                drawPoints(img, supportVectors, classesTrain, getRanges(featuresTest), classColors, 2);
                //svm = 0;
            }
            //*/
            // Выводим изображение в новом окне
            namedWindow(winName);
            imshow(winName, img);
        }
        //delete model;
        //model = 0;
        predictFunction = 0;

        printf("Do you want to continue? ESC - exit\n");
        // Ожидаем нажатия клавиши в графическом окне.
        ans = waitKey();
    }
    while (ans != escCode);

    // Закрываем все графические окна.
    destroyAllWindows();
    return 0;
}