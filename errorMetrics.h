#ifndef ERROR_METRICS_H
#define ERROR_METRICS_H

#include <opencv2/core/core.hpp>
#include "opencv2/ml/ml.hpp"
#include "auxilary.h"

/*
// Функция вычисления ошибки классификации.
// 
// API
// float getClassificationError(const Mat & samples,
//                              const Mat & classes,
//                              const cv::ml::StatModel & model,
//                              int (*predict) (const cv::Mat & sample,
                                      const cv::ml::StatModel & model))
// ВЫХОД
// samples    - матрица входных признаков прецедентов выборки.
// classes    - классы прецедентов выборки
//              (для проверки правильности предсказания).
// model      - обученная модель.
// predict    - функция предсказания для данной модели.
// 
*/
float getClassificationError(const cv::Mat & samples,
	const cv::Mat & classes,
	cv::Ptr<cv::ml::StatModel> model,
	bool test,
	//int (*predict) (const cv::Mat & sample,
	  //       cv::ml::StatModel & model));
	PredictionFunction * func);

#endif