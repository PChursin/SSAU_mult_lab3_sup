#include "cvgbtrees.h"

using namespace cv;


/*
// Функция обучения градиентного бустинга
// деревьев решений.
// 
// API
// void trainGBTrees(const cv::Mat & trainSamples,
//                  const cv::Mat & trainClasses,
//                  const CvGBTreesParams & params,
//                  CvGBTrees & gbtrees)
// 
// ВХОД
// trainSamples  - матрица входных признаков для прецедентов
//                 обучающей выборки
// trainClasses  - классы прецедентов обучающей выборки
// params        - параметры алгоритма обучения
// 
// ВХОД
// gbtrees        - обученная модель градиентного бустинга
*/
void trainGBTrees(const cv::Mat & trainSamples,
	const cv::Mat & trainClasses,
	Ptr<ml::Boost> boost)
                //const CvGBTreesParams & params,
                //CvGBTrees & gbtrees)
{
    /* =================================================================== */
    //  Напишите код, запускающий алгоритм обучения градиентного бустинга
    //  деревьев решений.
    //  Для обучения должны использоваться все прецеденты обучающей выборки
    //  и все признаки, пропущенных значений нет, все признаки кроме целевого
    //  количественные.
    /* ------------------------------------------------------------------- */
	Ptr<ml::TrainData> trainData = ml::TrainData::create(trainSamples, ml::ROW_SAMPLE, trainClasses);
	boost->train(trainData);
    /* =================================================================== */
}

/*
// Функция чтения параметров алгоритма обучения градиентного
// бустинга деревьев решений с консоли.
// 
// API
// CvGBTreesParams readGBTreesParams()
// 
// РЕЗУЛЬТАТ
// Параметры алгоритма обучения градиентного бустинга
// деревьев решений.
// 
*/
Ptr<ml::Boost> readGBTreesParams()
{
    int treeDepth = -1;
    printf("max depth of each tree = ");
    scanf("%d", &(treeDepth));

    int treesNum = -1;
    printf("number of trees to build = ");
    scanf("%d", &(treesNum));
	/*
    float learningRate = 0.0f;
    printf("learning rate (shrinkage) = ");
    scanf("%f", &(learningRate));

    float subsamplePortion = 0.0f;
    printf("subsample portion = ");
    scanf("%f", &(subsamplePortion));*/
    
	Ptr<ml::Boost> boost = ml::Boost::create();
	boost->setMaxDepth(treeDepth);
	boost->setWeakCount(treesNum);
	//boost->setBoostType(ml::Boost::Types::REAL);
	boost->setUseSurrogates(false);
	//boost->set
	/*
    CvGBTreesParams params(CvGBTrees::DEVIANCE_LOSS,
        treesNum,
        learningRate,
        subsamplePortion,
        treeDepth,
        false);*/

    return boost;
}
