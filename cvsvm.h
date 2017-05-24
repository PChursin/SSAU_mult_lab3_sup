#ifndef SUPPORT_VECTOR_MACHINE
#define SUPPORT_VECTOR_MACHINE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

/*
// ������� ������ ���������� ��������� �������� SVM � �������.
// 
// API
// CvSVMParams readSVMParams()
// 
// ���������
// ��������� ��������� �������� SVM.
// 
*/
cv::Ptr<cv::ml::SVM> readSVMParams(bool autoParams);
//CvSVMParams readSVMParams();



/*
// ������� �������� SVM-������.
// 
// API
// void trainSVM(const cv::Mat & trainSamples,
//               const cv::Mat & trainClasses,
//               const CvSVMParams & params,
//               CvSVM & svm)
// 
// ����
// trainSamples  - ������� ������� ��������� ��� �����������
//                 ��������� �������
// trainClasses  - ������ ����������� ��������� �������
// params        - ��������� ��������� ��������
// 
// ����
// svm           - ��������� SVM-������
*/
void trainSVM(const cv::Mat & trainSamples,
              const cv::Mat & trainClasses,
              //const CvSVMParams & params,
              cv::Ptr<cv::ml::SVM> svm);


/*
// ������� ������������ SVM-������.
// 
// API
// int getSVMPrediction(const cv::Mat & sample,
//                      const cv::ml::StatModel & model)
// 
// ����
// sample  - �������, ���������� ���������� ����� �����
//           � ������������ ���������
// model   - ��������� SVM-������
// 
// ���������
// ������������� �����
*/
int getSVMPrediction(const cv::Mat & sample,
	cv::ml::SVM & model);



cv::Mat getSupportVectors(const cv::Ptr<cv::ml::SVM> svm);



#endif