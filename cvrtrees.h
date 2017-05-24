#ifndef RANDOM_FOREST_TREE
#define RANDOM_FOREST_TREE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;
using namespace cv::ml;

/*
// Функция чтения параметров алгоритма обучения случайного леса с консоли.
// 
// API
// CvRTParams readRTreesParams()
// 
// РЕЗУЛЬТАТ
// Параметры алгоритма обучения случайного леса.
// 
*/
Ptr<RTrees> readRTreesParams();


/*
// Функция обучения случайного леса.
// 
// API
// void trainRTrees(const cv::Mat & trainSamples,
//                  const cv::Mat & trainClasses,
//                  const CvRTParams & params,
//                  CvRTrees & rtrees)
// 
// ВХОД
// trainSamples  - матрица входных признаков для прецедентов
//                 обучающей выборки
// trainClasses  - классы прецедентов обучающей выборки
// params        - параметры алгоритма обучения
// 
// ВХОД
// rtrees        - обученный случайный лес
*/
void trainRTrees(const cv::Mat & trainSamples,
                const cv::Mat & trainClasses,
				Ptr<RTrees> rtrees);
                //const CvRTParams & params,
                //CvRTrees & rtrees);



#endif