#ifndef DRAWING_FUNCTIONS
#define DRAWING_FUNCTIONS

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml/ml.hpp>
#include <map>

/*
// Тип данных: ФУНКЦИЯ getPredictedClassLabel
// Определяет общий вид функции, осуществляющей предсказание
// с помощью обученного классификатора.
//
// ВХОД
// sample       - матрица, содержащая координаты
//                одной точки в пространстве признаков
// model        - обученная модель
// РЕЗУЛЬТАТ
// Предсказанная метка класса.
*/
/*typedef int (getPredictedClassLabel) (const cv::Mat & sample,
                                      const cv::ml::StatModel & model);
									  */
typedef int (getPredictedClassLabel) (const cv::Mat & sample,
                                      const cv::Ptr<cv::ml::SVM> & model);
/*
// Функция рисования областей, на которые
// разбивает пространство признаков классификатор
// 
// API
// drawPartition(cv::Mat & img,
//               std::map<int, cv::Scalar> & classColors,
//               const cv::Mat & dataRanges,
//               const cv::Size stepsNum,
//               const cv::ml::StatModel & model,
//               getPredictedClassLabel * predictLabel)
// ВХОД
// classColors       - соотвествие между метками классов и цветами.
//                     Если для определенного класса не задано соответствие,
//                     будет добавлен случайный цвет.
// dataRanges        - Определяет границы области рисования в координатах
//                     пространства признаков. Матрица 2x2: первый столбец
//                     содержит max и min первой переменной, второй -- второй.
// stepsNum          - количество шагов по каждой координате.
// model             - обученный классификатор
// predictLabel      - указатель на функцию предсказания для используемого
//                     классификатора
// 
// ВЫХОД
// img               - изображение с отрисованными областями
*/
void drawPartition(cv::Mat & img,
                   std::map<int, cv::Scalar> & classColors,
                   const cv::Mat & dataRanges,
                   const cv::Size stepsNum,
                   const cv::Ptr<cv::ml::SVM> & model,
                   getPredictedClassLabel * predictLabel);


/*
// Функция рисования точек двумерного пространства признаков
// 
// API
// drawPoints(cv::Mat & img,
//            const cv::Mat & data,
//            const cv::Mat & classes,
//            const cv::Mat & ranges,
//            std::map<int, cv::Scalar> & classColors,
//            int drawingMode = 0);
// ВХОД
// data              - матрица, в каждой строке которой координаты одной точки
//                     в признаковом пространстве
// classes           - метки классов точек
// ranges            - Определяет границы области рисования в координатах
//                     пространства признаков. Матрица 2x2: первый столбец содержит
//                     max и min первой переменной, второй -- второй.
// classColors       - соотвествие между метками классов и цветами.
//                     Если для определенного класса не задано соответствие,
//                     будет добавлен случайный цвет.
// drawingMode       - тип отрисовки точек: 0 - круг,
//                     1 - окружность, 2 - черная точка.
//
// ВЫХОД
// img               - изображение с отрисованными точками
*/
void drawPoints(cv::Mat & img,
                const cv::Mat & data,
                const cv::Mat & classes,
                const cv::Mat & ranges,
                std::map<int, cv::Scalar> & classColors,
                int drawingMode = 0);


/*
// Функция получения максимальных и минимальных значений каждой переменной
// 
// API
// cv::Mat getRanges(const cv::Mat & data);
//
// ВХОД
// data              - матрица, в каждой строке которой координаты одной точки
//                     в признаковом пространстве
//
// РЕЗУЛЬТАТ
// Матрица 2x<кол-во переменных>: в каждом столбце max и min
// для соответствующей переменной.
*/
cv::Mat getRanges(const cv::Mat & data);

#endif