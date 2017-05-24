#ifndef DECISION_TREE
#define DECISION_TREE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;

/*
// Функция чтения параметров алгоритма обучения дерева решений с консоли.
// 
// API
// CvSVMParams readSVMParams()
// 
// РЕЗУЛЬТАТ
// Параметры алгоритма обучения дерева решений.
// 
*/
Ptr<ml::DTrees> readDTreeParams();


/*
// Функция обучения дерева решений.
// 
// API
// void trainDTree(const cv::Mat & trainSamples,
//                 const cv::Mat & trainClasses,
//                 const CvDTreeParams & params,
//                 CvDTree & dtree)
// 
// ВХОД
// trainSamples  - матрица входных признаков для прецедентов
//                 обучающей выборки
// trainClasses  - классы прецедентов обучающей выборки
// params        - параметры алгоритма обучения
// 
// ВХОД
// dtree         - обученное дерево решений
*/
void trainDTree(const cv::Mat & trainSamples,
	const cv::Mat & trainClasses,
	const cv::Ptr<ml::DTrees> params);
                //CvDTree & dtree);

#endif