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
	bool test)
{

	Mat out;
	Ptr<ml::TrainData> trainData = ml::TrainData::create(samples, ml::ROW_SAMPLE, classes);
	int nClasses = trainData->getClassLabels().rows;

	int** errors = new int*[nClasses];
	for (int i = 0; i < nClasses; i++) {
		errors[i] = new int[nClasses];
		memset(errors[i], 0, nClasses*sizeof(int));
	}

	float tmpErr = model->calcError(trainData, test, out);

	for (int i = 0; i < out.rows; i++) {
		errors[(int)out.at<float>(i)][classes.at<int>(i)]++;
	}
	printf("Classification matrix for %s dataset\n", test ? "train" : "test");
	for (int i = 0; i < nClasses; i++) {
		for (int j = 0; j < nClasses; j++) {
			printf("%4d ", errors[i][j]);
		}
		printf("\n");
	}

	for (int i = 0; i < nClasses; i++)
		delete[] errors[i];
	delete errors;
    return tmpErr;
}