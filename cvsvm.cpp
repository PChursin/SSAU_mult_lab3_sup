#include <stdio.h>
#include "cvsvm.h"

#pragma warning(disable : 4996)

using namespace cv;


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
              Ptr<ml::SVM> svm)
{
    /* =================================================================== */
    //  Напишите код, запускающий алгоритм обучения SVM-классификатора.
    //  Для обучения должны использоваться все прецеденты обучающей выборки
    //  и все признаки.
    /* ------------------------------------------------------------------- */

	svm->train(trainSamples, ml::ROW_SAMPLE, trainClasses);



    /* =================================================================== */
}


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
int getSVMPrediction(const Mat & sample,
                     ml::SVM & model)
{
    //const CvSVM & svm = dynamic_cast<const CvSVM &>(model);
    int prediction = 0;
    /* =================================================================== */
    //  Напишите код, запускающий алгоритм предсказания SVM-классификатора.
    /* ------------------------------------------------------------------- */
	//std::vector<Mat> in;
	//in.push_back(sample);
	Mat out;
	model.predict(sample, out);
	prediction = out.at<int>(0);



    /* =================================================================== */
    return prediction;
}


Mat getSupportVectors(const Ptr<ml::SVM> svm)
{
    //Mat sv;
    /* =================================================================== */
    //  Напишите код, построчно заполняющий матрицу sv типа CV_32F
    //  опорными векторами модели svm.
    /* ------------------------------------------------------------------- */
	return svm->getSupportVectors();




    /* =================================================================== */
    //return sv;
}


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
Ptr<ml::SVM> readSVMParams(bool autoParams)
{
	Ptr<cv::ml::SVM> params = cv::ml::SVM::create();
	params->setType(ml::SVM::C_SVC);

    // Устанавливаем параметры остановки итерационного метода оптимизации
    params->setTermCriteria(
        cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000000, 0.0001));
	if (autoParams)
		return params;
	
    // Запрашиваем параметр стоимости неправильной классификации
    printf("C = ");
	double cParam;
    scanf("%lf", &cParam);
	params->setC(cParam);

    // Выбираем ядро
	int kType;
    do
    {
        printf("Choose kernel:\n\t0 - Linear kernel K(u, v) = <u, v>\n");
        printf("\t1 - Polinomial kernel K(u, v) = (gamma * <u, v> + coef0) ^ degree\n");
        printf("\t2 - RBF kernel K(u, v) = exp(-gamma * ||u - v|| ^ 2)\n");
        printf("\t3 - Sigmoid kernel K(u, v) = tanh(gamma * <u, v> + coef0)\n");
        printf("kernel type = ");		
        scanf("%d", &kType);
    }
    while (kType < 0 || kType > 3);
	params->setKernel(kType);
    
    // Выбираем параметры ядра
	double gamma, coef0, degree;
    switch (kType)
    {
	case ml::SVM::LINEAR: 
        break;
    case ml::SVM::POLY: 
        {
            printf("gamma = ");
            scanf("%lf", &gamma);
            printf("coef0 = ");
            scanf("%lf", &coef0);
            printf("degree = ");
            scanf("%lf", &degree);
			params->setGamma(gamma);
			params->setCoef0(coef0);
			params->setDegree(degree);
        }; break;
    case ml::SVM::RBF: 
        {
            printf("gamma = ");
            scanf("%lf", &gamma);
			params->setGamma(gamma);
        }; break;
    case ml::SVM::SIGMOID: 
        {
            printf("gamma = ");
            scanf("%lf", &gamma);
            printf("coef0 = ");
            scanf("%lf", &coef0);
			params->setGamma(gamma);
			params->setCoef0(coef0);
        }; break;
    }
	

    return params;
}

