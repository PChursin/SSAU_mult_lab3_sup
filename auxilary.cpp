#include "auxilary.h"

using namespace cv;
using std::map;


/*
// ������� ��������� ����� ���������� ������������ ���������
// 
// API
// drawPoints(cv::Mat & img,
//            const cv::Mat & data,
//            const cv::Mat & classes,
//            const cv::Mat & ranges,
//            std::map<int, cv::Scalar> & classColors,
//            int drawingMode = 0);
// ����
// data              - �������, � ������ ������ ������� ���������� ����� �����
//                     � ����������� ������������
// classes           - ����� ������� �����
// ranges            - ���������� ������� ������� ��������� � �����������
//                     ������������ ���������. ������� 2x2: ������ ������� ��������
//                     max � min ������ ����������, ������ -- ������.
// classColors       - ����������� ����� ������� ������� � �������.
//                     ���� ��� ������������� ������ �� ������ ������������,
//                     ����� �������� ��������� ����.
// drawingMode       - ��� ��������� �����: 0 - ����,
//                     1 - ����������, 2 - ������ �����.
//
// �����
// img               - ����������� � ������������� �������
*/
void drawPoints(Mat & img,
                const Mat & data,
                const Mat & classes,
                const Mat & ranges,
                map<int, Scalar> & classColors,
                int drawingMode)
{
    if (data.cols != 2 || ranges.cols != 2)
    {
        printf("warning. only two-dimensional feature spaces can be visualized.\n");
        return;
    }
    // scale points
    int samplesNum = data.rows;
    int varsNum = data.cols;
    assert(varsNum == 2);

    const int border = 8;
    Mat intPoints(samplesNum, varsNum, CV_32S);
    double scaleX =  (double)(img.cols - 2 * border) / (ranges.at<double>(1, 0) - ranges.at<double>(0, 0));
    data.col(0).convertTo(intPoints.col(0), CV_32S, scaleX, border - ranges.at<double>(0, 0) * scaleX);
    double scaleY = (double)(img.rows - 2 * border) / (ranges.at<double>(1, 1) - ranges.at<double>(0, 1));
    data.col(1).convertTo(intPoints.col(1), CV_32S, scaleY, border - ranges.at<double>(0, 1) * scaleY);

    // draw points
    RNG rng = theRNG();
    for (int i = 0; i < samplesNum; ++i)
    {
        int classLabel = classes.at<int>(i);
        if (classColors.count(classLabel) == 0)
        {
            classColors[classLabel] = Scalar((uchar)rng(256), (uchar)rng(256), (uchar)rng(256));
        }
        switch (drawingMode)
        {
        case 0:
            {
                circle(img,
                    Point(intPoints.at<int>(i,0),
                    img.rows - intPoints.at<int>(i,1) - 1),
                    4,
                    classColors[classLabel], -1);
            }; break;
        case 1:
            {
                circle(img,
                    Point(intPoints.at<int>(i,0),
                    img.rows - intPoints.at<int>(i,1) - 1),
                    3,
                    classColors[classLabel],
                    2);
                circle(img,
                    Point(intPoints.at<int>(i,0),
                    img.rows - intPoints.at<int>(i,1) - 1),
                    2,
                    Scalar(255,255,255),
                    -1);
            }; break;
        case 2:
            {
                circle(img,
                    Point(intPoints.at<int>(i,0),
                    img.rows - intPoints.at<int>(i,1) - 1),
                    2,
                    Scalar(0, 0, 0),
                    -1);
            }; break;
        }
    }
}


/*
// ������� ��������� ��������, �� ������� ���������
// ������������ ��������� �������������
// 
// API
// drawPartition(cv::Mat & img,
//               std::map<int, cv::Scalar> & classColors,
//               const cv::Mat & dataRanges,
//               const cv::Size stepsNum,
//               const cv::ml::StatModel & model,
//               getPredictedClassLabel * predictLabel)
// ����
// classColors       - ����������� ����� ������� ������� � �������.
//                     ���� ��� ������������� ������ �� ������ ������������,
//                     ����� �������� ��������� ����.
// dataRanges        - ���������� ������� ������� ��������� � �����������
//                     ������������ ���������. ������� 2x2: ������ ������� ��������
//                     max � min ������ ����������, ������ -- ������.
// stepsNum          - ���������� ����� �� ������ ����������.
// model             - ��������� �������������
// predictLabel      - ��������� �� ������� ������������ ��� ������������� ��������������
// 
// �����
// img               - ����������� � ������������� ���������
*/
void drawPartition(cv::Mat & img,
					std::map<int, cv::Scalar> & classColors,
					const cv::Mat & dataRanges,
					const cv::Size stepsNum,
					cv::Ptr<cv::ml::StatModel> model,
					//getPredictedClassLabel * predictLabel)
					PredictionFunction * predictLabel)
{
    if (dataRanges.cols != 2)
    {
        printf("warning. only two-dimensional feature spaces can be visualized.\n");
        return;
    }
    const int border = 8;

    double rangeX = (dataRanges.at<double>(1, 0) - dataRanges.at<double>(0, 0));
    double rangeY = (dataRanges.at<double>(1, 1) - dataRanges.at<double>(0, 1));
    double scaleX = (double)(img.cols - 2 * border) / rangeX;
    double scaleY = (double)(img.rows - 2 * border) / rangeY;
    float stepX = (float)(rangeX) / (float)(stepsNum.width);
    float stepY = (float)(rangeY) / (float)(stepsNum.height);

    Mat sample(1, 2, CV_32F);
    for (int i = 0; i < stepsNum.width; ++i)
    {
        for (int j = 0; j < stepsNum.height; ++j)
        {
            float x = (float)(dataRanges.at<double>(0, 0)) + i * stepX;
            float y = (float)(dataRanges.at<double>(0, 1)) + j * stepY;
            sample.at<float>(0) = x;
            sample.at<float>(1) = y;

            //int prediction = (*predictLabel)(sample, model);
			int prediction = predictLabel->predict(sample, *(model.get()));
            
            int xImg = (int)((x - dataRanges.at<double>(0, 0)) * scaleX) +
                border;
            int yImg = img.rows -
                (int)((y - dataRanges.at<double>(0, 1)) * scaleY) -
                border - 1;

            if (classColors.count(prediction) == 0)
            {
                RNG & rng = theRNG();
                classColors[prediction] =
                    Scalar((uchar)rng(256), (uchar)rng(256), (uchar)rng(256));
            }
            Scalar color = classColors[prediction];
            circle(img, Point(xImg, yImg), 1, color);
        }
    }
}


/*
// ������� ��������� ������������ � ����������� �������� ������ ����������
// 
// API
// cv::Mat getRanges(const cv::Mat & data);
//
// ����
// data              - �������, � ������ ������ ������� ���������� ����� �����
//                     � ����������� ������������
//
// ���������
// ������� 2x<���-�� ����������>: � ������ ������� max � min
// ��� ��������������� ����������.
*/
Mat getRanges(const Mat & data)
{
    Mat minMaxValues(2, data.cols, CV_64F);
    for (int i = 0; i < data.cols; ++i)
    {
        minMaxLoc(data.col(i),
            &(minMaxValues.at<double>(0, i)),
            &(minMaxValues.at<double>(1, i)));
    }

    return minMaxValues;
}


/*
// ������� ������ ��������� � �������� ������� �� YAML-�����.
// 
// API
// void readDatasetFromFile(Mat & featuresTrain,
//                          Mat & classesTrain,
//                          Mat & featuresTest,
//                          Mat & classesTest)
// �����
// featuresTrain - ������� ������� ��������� ��� �����������
//                 ��������� �������
// classesTrain  - ������ ����������� ��������� �������
// featuresTest  - ������� ������� ��������� ��� �����������
//                 �������� �������
// classesTest   - ������ ����������� �������� �������
// 
*/
void readDatasetFromFile(Mat & featuresTrain,
                         Mat & classesTrain,
                         Mat & featuresTest,
                         Mat & classesTest)
{
    char fileName[maxFileNameLen];
    do 
    {
        printf("File name = ");
        scanf("%s", &fileName);
        FileStorage fs;
        fs.open(fileName, FileStorage::READ);
        if (fs.isOpened() == false)
        {
            continue;
        }
        // ������ �������. ���� � ����� ��� ����� � ������ ����������
        // ������� ������ �������.
        fs["features_train"] >> featuresTrain;
        fs["response_train"] >> classesTrain;
        fs["features_test"] >> featuresTest;
        fs["response_test"] >> classesTest;
    }
    // ��������� ���� �� ������� ���������� ����.
    while (featuresTrain.empty()
        || classesTrain.empty()
        || featuresTest.empty()
        || classesTest.empty());
}

int SVMPrediction::predict(const cv::Mat & sample, cv::ml::StatModel & model)
{
	int prediction = 0;

	Mat out;
	model.predict(sample, out);
	prediction = out.at<int>(0);

	return prediction;
}

int DTreePrediction::predict(const cv::Mat & sample, cv::ml::StatModel & model)
{
	int prediction = 0;

	Mat out;
	model.predict(sample, out);
	prediction = out.at<int>(0);

	return prediction;
}
