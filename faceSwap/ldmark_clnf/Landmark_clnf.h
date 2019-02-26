#ifndef __FACE_LANDMARK_CLNF_h_
#define __FACE_LANDMARK_CLNF_h_
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "LandmarkDetectorFunc.h"
using namespace std;
using namespace cv;

#define  FACE_RECT_X   100
#define  FACE_RECT_Y  100
#define  FACE_RECT_W   291
#define  FACE_RECT_H  301
#define  FACE_EYE_DIST  140

struct  shapeCaches
{
	cv::Vec3d rotation_hypothese;
	vector<cv::Point>  landmark5;
};

class Landmark_clnf
{
public:
	Landmark_clnf();
	~Landmark_clnf();
	int init(string modelDir);

	//根据mtcnn的5个点计算人脸的初始shape
	void getInitShape(cv::Mat& img, cv::Rect& mtcnn_faceBox, vector<cv::Point>& mtcnn_ldmark, vector<cv::Point>& init_landmark);

	vector<vector<cv::Point> >  LandmarkImage(cv::Mat& img, vector<cv::Rect>& mtcnn_faceBoxes, vector<vector<cv::Point> >& mtcnn_landmarks);
		
private:
	LandmarkDetector::FaceModelParameters landmark_param;
	LandmarkDetector::CLNF landmark_model;	
	vector<cv::Vec3d> rotation_hypotheses_inits;
	vector<shapeCaches>  m_shapeCaches;
	void  conv_landmark68_2_landmark5(vector<cv::Point>& landmark5, vector<cv::Point>& landmark68);
	void  generateEuler(vector<cv::Vec3d>& rotation_hypothese);
	double  get_5points_dis(vector<cv::Point>& landmark1, vector<cv::Point>& landmark2);
	
	void  use_pdm_get_initShape(cv::Rect& face, LandmarkDetector::CLNF& landmark_model, cv::Vec3d& rotation_hypothese, vector<cv::Point>& landmark);
	void  refineMtcnnBox(cv::Rect& box, vector<cv::Point>& landmarks, int faceDetectType);

	void  initShapes(LandmarkDetector::CLNF& landmark_model);
	void  refineModelLdmark(vector<cv::Point>& landmark5);
	void getCurLdmarkCof(cv::Rect& face, vector<cv::Point>& landmark5, float& offset_x, float& offset_y, float& scale);
	void  refineLdmark(cv::Mat& img, cv::Rect& face, vector<cv::Point>& landmark5);
	void  clc_shape_pdm(cv::Mat& img, cv::Rect& face, vector<cv::Point>& landmark5, int faceDetectType, vector<cv::Vec3d>& rotation_hypothese);
};

#endif
