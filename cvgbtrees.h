#ifndef GRADIENT_BOOSTED_TREES
#define GRADIENT_BOOSTED_TREES

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"


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
CvGBTreesParams readGBTreesParams();


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
                const CvGBTreesParams & params,
                CvGBTrees & gbtrees);


/*
// Функция предсказания с помощью модель градиентного бустинга
// деревьев решений.
// 
// API
// int getGBTreesPrediction(const cv::Mat & sample,
//                         const cv::ml::StatModel & model)
// 
// ВХОД
// sample  - матрица, содержащая координаты одной точки
//           в пространстве признаков
// model   - обученная модель градиентного бустинга
// 
// РЕЗУЛЬТАТ
// Предсказанный класс
*/
int getGBTreesPrediction(const cv::Mat & sample,
                        const cv::ml::StatModel & model);


#endif