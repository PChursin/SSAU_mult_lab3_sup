#ifndef SUPPORT_VECTOR_MACHINE
#define SUPPORT_VECTOR_MACHINE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

/*
// Функция чтения параметров алгоритма обучения SVM с консоли.
// 
// API
// CvSVMParams readSVMParams()
// 
// РЕЗУЛЬТАТ
// Параметры алгоритма обучения SVM.
// 
*/
cv::Ptr<cv::ml::SVM> readSVMParams(bool autoParams);
//CvSVMParams readSVMParams();



/*
// Функция обучения SVM-модели.
// 
// API
// void trainSVM(const cv::Mat & trainSamples,
//               const cv::Mat & trainClasses,
//               const CvSVMParams & params,
//               CvSVM & svm)
// 
// ВХОД
// trainSamples  - матрица входных признаков для прецедентов
//                 обучающей выборки
// trainClasses  - классы прецедентов обучающей выборки
// params        - параметры алгоритма обучения
// 
// ВХОД
// svm           - обученная SVM-модель
*/
void trainSVM(const cv::Mat & trainSamples,
              const cv::Mat & trainClasses,
              //const CvSVMParams & params,
              cv::Ptr<cv::ml::SVM> svm);


/*
// Функция предсказания SVM-модели.
// 
// API
// int getSVMPrediction(const cv::Mat & sample,
//                      const cv::ml::StatModel & model)
// 
// ВХОД
// sample  - матрица, содержащая координаты одной точки
//           в пространстве признаков
// model   - обученная SVM-модель
// 
// РЕЗУЛЬТАТ
// Предсказанный класс
*/
int getSVMPrediction(const cv::Mat & sample,
	cv::ml::SVM & model);



cv::Mat getSupportVectors(const cv::Ptr<cv::ml::SVM> svm);



#endif