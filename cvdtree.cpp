#include "cvdtree.h"

using namespace cv;


/*
// Функция обучения дерева решений.
// 
// API
// void trainDTree(const cv::Mat & trainSamples,
//                 const cv::Mat & trainClasses,
//                 const CvDTreeParams & params,
//                 CvDTree & dtree)
// 
// ВХОД
// trainSamples  - матрица входных признаков для прецедентов
//                 обучающей выборки
// trainClasses  - классы прецедентов обучающей выборки
// params        - параметры алгоритма обучения
// 
// ВХОД
// dtree         - обученное дерево решений
*/
void trainDTree(const Mat & trainSamples,
                const Mat & trainClasses,
                //const CvDTreeParams & params,
                //CvDTree & dtree)
				Ptr<ml::DTrees> dtree)
{
    /* =================================================================== */
    //  Напишите код, запускающий алгоритм обучения дерева решений.
    //  Для обучения должны использоваться все прецеденты обучающей выборки
    //  и все признаки, пропущенных значений нет, все признаки кроме целевого
    //  количественные.
    /* ------------------------------------------------------------------- */
	Ptr<ml::TrainData> trainData = ml::TrainData::create(trainSamples, ml::ROW_SAMPLE, trainClasses);
	dtree->train(trainData);






    /* =================================================================== */
}


/*
// Функция предсказания с помощью дерева решений.
// 
// API
// int getDTreePrediction(const cv::Mat & sample,
//                        const cv::ml::StatModel & model)
// 
// ВХОД
// sample  - матрица, содержащая координаты одной точки
//           в пространстве признаков
// model   - обученное дерево решений
// 
// РЕЗУЛЬТАТ
// Предсказанный класс
*/
int getDTreePrediction(const Mat & sample, const cv::ml::StatModel & model)
{
    // Приводим тип стат. модели к дереву решений
    //const CvDTree & dtree = dynamic_cast<const CvDTree &>(model);
    int prediction = 0;
    /* =================================================================== */
    //  Напишите код, запускающий алгоритм предсказания для дерева решений.
    /* ------------------------------------------------------------------- */

	Mat out;
	model.predict(sample, out);
	prediction = out.at<int>(0);




    /* =================================================================== */
    return prediction;
}


/*
// Функция чтения параметров алгоритма обучения дерева решений с консоли.
// 
// API
// CvSVMParams readSVMParams()
// 
// РЕЗУЛЬТАТ
// Параметры алгоритма обучения дерева решений.
// 
*/
Ptr<ml::DTrees> readDTreeParams()
{
	Ptr<ml::DTrees> params = ml::DTrees::create();
	
    //CvDTreeParams params;
    
    // Не строим суррогатные разбиения.
    params->setUseSurrogates(false);
    // Не храним поддеревья, удаленные в ходе обрезки дерева (pruning).
    params->setTruncatePrunedTree(true);
    // Параметр алгоритма обрезания дерева.
    params->setUse1SERule(true);

    // Максимальная высота (глубина) дерева.
	int tmp;
    printf("maximal tree depth = ");
    scanf("%d", &tmp);
	params->setMaxDepth(tmp);

    // Минимальное количество прецедентов обучающей выборке,
    // при котором выполняется дальнейшее разбиение.
    printf("minimal number of samples in leaf = ");
    scanf("%d", &tmp);
	params->setMinSampleCount(tmp);


    // Количество блоков в скользящем контроле (cross-validation).
    int doPruning = 0;
    printf("apply pruning (0/1) = ");
    scanf("%d", &(doPruning));
    // Т.к. данных в тестовых задачах немного,
    // используем пятикратный скользящий контроль.
    params->setCVFolds((doPruning == 0) ? 0 : 5);

    return params;
}

