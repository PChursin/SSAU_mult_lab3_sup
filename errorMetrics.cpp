#include "errorMetrics.h"
#include "auxilary.h"

using namespace cv;
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
	//int(*predict) (const cv::Mat & sample,
	//	cv::ml::StatModel & model))
	PredictionFunction * func)
{
    float error = 0.0f;
    /* =================================================================== */
    //  �������� ��� ���������� ������ �������������.
    //  ������ ������������ ������������� �����������
    //  ��� ��������� ���-�� ��������� ������������������
    //  �������� ������� � ������ ���-�� �������� � �������.
    //  ��� ��������� ������������ ��������� ���������
    //  ����� (*predict)(sample, model), ����������� �������� �����
    //  ����� �������������� ������.
    /* ------------------------------------------------------------------- */
	Mat out;
	Ptr<ml::TrainData> trainData = ml::TrainData::create(samples, ml::ROW_SAMPLE, classes);

	model->calcError(trainData, test, out);

	for (int i = 0; i < out.size().area(); i++) {
		error += out.at<float>(i);
	}

	error /= out.size().area();




    /* =================================================================== */
    return error;
}