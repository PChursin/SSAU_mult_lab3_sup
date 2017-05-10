#include "errorMetrics.h"
#include "auxilary.h"

using namespace cv;
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
	//int(*predict) (const cv::Mat & sample,
	//	cv::ml::StatModel & model))
	PredictionFunction * func)
{
    float error = 0.0f;
    /* =================================================================== */
    //  Напишите код вычисления ошибки классификации.
    //  Ошибка неправильной классификации вычисляется
    //  как отношение кол-ва правильно классифицированных
    //  объектов выборки к общему кол-ву объектов в выборке.
    //  Для получения предсказания требуется выполнить
    //  вызов (*predict)(sample, model), результатов которого будет
    //  номер предсказанного класса.
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