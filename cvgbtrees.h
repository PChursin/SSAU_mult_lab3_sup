#ifndef GRADIENT_BOOSTED_TREES
#define GRADIENT_BOOSTED_TREES

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;


/*
// Функция чтения параметров алгоритма обучения градиентного
// бустинга деревьев решений с консоли.
// 
// API
// CvGBTreesParams readGBTreesParams()
// 
// РЕЗУЛЬТАТ
// Параметры алгоритма обучения градиентного бустинга
// деревьев решений.
// 
*/
Ptr<ml::Boost> readGBTreesParams();


/*
// Функция обучения градиентного бустинга
// деревьев решений.
// 
// API
// void trainGBTrees(const cv::Mat & trainSamples,
//                  const cv::Mat & trainClasses,
//                  const CvGBTreesParams & params,
//                  CvGBTrees & gbtrees)
// 
// ВХОД
// trainSamples  - матрица входных признаков для прецедентов
//                 обучающей выборки
// trainClasses  - классы прецедентов обучающей выборки
// params        - параметры алгоритма обучения
// 
// ВХОД
// gbtrees        - обученная модель градиентного бустинга
*/
void trainGBTrees(const cv::Mat & trainSamples,
	const cv::Mat & trainClasses,
	Ptr<ml::Boost> boost);
                //const CvGBTreesParams & params,
                //CvGBTrees & gbtrees);



#endif