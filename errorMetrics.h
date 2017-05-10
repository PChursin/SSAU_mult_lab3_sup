#ifndef ERROR_METRICS_H
#define ERROR_METRICS_H

#include <opencv2/core/core.hpp>
#include "opencv2/ml/ml.hpp"
#include "auxilary.h"

/*
// ������� ���������� ������ �������������.
// 
// API
// float getClassificationError(const Mat & samples,
//                              const Mat & classes,
//                              const cv::ml::StatModel & model,
//                              int (*predict) (const cv::Mat & sample,
                                      const cv::ml::StatModel & model))
// �����
// samples    - ������� ������� ��������� ����������� �������.
// classes    - ������ ����������� �������
//              (��� �������� ������������ ������������).
// model      - ��������� ������.
// predict    - ������� ������������ ��� ������ ������.
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