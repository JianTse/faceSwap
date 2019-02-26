// faceSwap.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "mtcnn_facedetect\mtcnn_opencv.h"
#include "ldmark_clnf\Landmark_clnf.h"
#include "swap\swap.h"

void  getImageLdmarks(MTCNN& detector, Landmark_clnf&  ldmarker, cv::Mat& img, vector<vector<cv::Point> >& ldmark68)
{
	float factor = 0.709f;
	float threshold[3] = { 0.7f, 0.6f, 0.6f };
	int minSize = 12;

	vector<FaceInfo> faceInfo = detector.Detect_mtcnn(img, minSize, threshold, factor, 3);

	vector<cv::Rect>mtcnn_faceRects;
	vector<vector<cv::Point> > mtcnn_landmarks;
	for (int i = 0; i < faceInfo.size(); i++) {
		int x = (int)faceInfo[i].bbox.xmin;
		int y = (int)faceInfo[i].bbox.ymin;
		int w = (int)(faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
		int h = (int)(faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
		cv::Rect faceRect = cv::Rect(x, y, w, h);
		mtcnn_faceRects.push_back(faceRect);

		vector<cv::Point> ldmark5;
		for (int k = 0; k < 5; k++)
		{
			int x = int(faceInfo[i].landmark[k * 2]);
			int y = int(faceInfo[i].landmark[k * 2 + 1]);
			cv::Point pt = cv::Point(x, y);
			ldmark5.push_back(pt);
		}
		mtcnn_landmarks.push_back(ldmark5);
	}
	ldmark68 = ldmarker.LandmarkImage(img, mtcnn_faceRects, mtcnn_landmarks);
}

void  drawLdmark(cv::Mat& img, vector<cv::Point>& ldmarks)
{
	for (int k = 0; k < ldmarks.size(); k++)
	{
		cv::circle(img, ldmarks[k], 1, cv::Scalar(0, 0, 255));
	}
}

int main(int argc, char **argv)
{
	MTCNN detector("model/mtcnn_model");

	Landmark_clnf  ldmarker;
	ldmarker.init("model/");

	faceSwap  _swaper;

	string  imgFn1 = "image/file0001.jpg";
	cv::Mat image1 = cv::imread(imgFn1, 1);

	string  imgFn2 = "image/file0011.jpg";
	cv::Mat image2 = cv::imread(imgFn2, 1);

	vector<vector<cv::Point> > ldmark1, ldmark2;
	getImageLdmarks(detector, ldmarker, image1, ldmark1);
	getImageLdmarks(detector, ldmarker, image2, ldmark2);

	cv::Mat dst;
	_swaper.process(image1, ldmark1[0], image2, ldmark2[0], dst);

	//画信息
	for (int i = 0; i < ldmark1.size(); i++) {
		drawLdmark(image1, ldmark1[i]);
	}
	for (int i = 0; i < ldmark2.size(); i++) {
		drawLdmark(image2, ldmark2[i]);
	}
	cv::imwrite("image/dst.jpg", dst);

	cv::imshow("image1", image1);
	cv::imshow("image2", image2);
	cv::imshow("dst", dst);
	cv::waitKey(0);


	return 1;
}


