#include "cvrtrees.h"

using namespace cv;


/*
// Функция обучения случайного леса.
// 
// API
// void trainRTrees(const cv::Mat & trainSamples,
//                  const cv::Mat & trainClasses,
//                  const CvRTParams & params,
//                  CvRTrees & rtrees)
// 
// ВХОД
// trainSamples  - матрица входных признаков для прецедентов
//                 обучающей выборки
// trainClasses  - классы прецедентов обучающей выборки
// params        - параметры алгоритма обучения
// 
// ВХОД
// rtrees        - обученный случайный лес
*/
void trainRTrees(const Mat & trainSamples,
                const Mat & trainClasses,
                const CvRTParams & params,
                CvRTrees & rtrees)
{
    /* =================================================================== */
    //  Напишите код, запускающий алгоритм обучения случайного леса.
    //  Для обучения должны использоваться все прецеденты обучающей выборки
    //  и все признаки, пропущенных значений нет, все признаки кроме целевого
    //  количественные.
    /* ------------------------------------------------------------------- */






    /* =================================================================== */
}


/*
// Функция предсказания с помощью случайного леса.
// 
// API
// int getRTreesPrediction(const cv::Mat & sample,
//                         const cv::ml::StatModel & model)
// 
// ВХОД
// sample  - матрица, содержащая координаты одной точки
//           в пространстве признаков
// model   - обученный случайный лес
// 
// РЕЗУЛЬТАТ
// Предсказанный класс
*/
int getRTreesPrediction(const Mat & sample,
                        const cv::ml::StatModel & model)
{
    // Приводим тип модели к случайному лесу
    const CvRTrees & rf = dynamic_cast<const CvRTrees &>(model);
    int prediction = 0;
    /* =================================================================== */
    //  Напишите код, запускающий алгоритм предсказания для случайного леса.
    /* ------------------------------------------------------------------- */






    /* =================================================================== */
    return prediction;
}


/*
// Функция чтения параметров алгоритма обучения случайного леса с консоли.
// 
// API
// CvRTParams readRTreesParams()
// 
// РЕЗУЛЬТАТ
// Параметры алгоритма обучения случайного леса.
// 
*/
CvRTParams readRTreesParams()
{
    int treeDepth = -1;
    printf("max depth of each tree = ");
    scanf("%d", &(treeDepth));

    int treesNum = -1;
    printf("number of trees to build = ");
    scanf("%d", &(treesNum));

    int activeVarsNum = -1;
    printf("number of active variables (set 0 to use the sqrt(total number of features)) = ");
    scanf("%d", &(activeVarsNum));
    
    CvRTParams params(treeDepth, 0, 0.0f, false, 10, 0, false, 0, treesNum, 0.0f, CV_TERMCRIT_ITER);

    return params;
}
