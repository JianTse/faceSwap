#include "Landmark_clnf.h"
#include "LandmarkDetectorFunc.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define CVV_PI   3.1415926535897932384626433832795

bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first < rhs.first;
}
/* Return the indices of the top N values of vector v. */
std::vector<int> Dismin(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result(N);
	for (int i = 0; i < N; ++i)
		result[i] = pairs[i].second;
	return result;
}

Landmark_clnf::Landmark_clnf()
{
	generateEuler(rotation_hypotheses_inits);
}
Landmark_clnf::~Landmark_clnf()
{

}

int Landmark_clnf::init(string modelDir)
{
	std::string main_location = "main_clnf_general.txt";
	landmark_model.init(modelDir, main_location);
	if (!landmark_model.loaded_successfully)
	{
		cout << "ERROR: Could not load the landmark detector" << endl;
		return 0;
	}

	//初始化respond
	landmark_model.initRespond(landmark_param);
	//执行一次
	cv::Mat img = cv::Mat::zeros(cv::Size(640,480), CV_8UC1);
	cv::Rect faceBox = cv::Rect(200, 100, 150,150);
	vector<cv::Vec3d> rotation_hypothese;
	rotation_hypothese.push_back(cv::Vec3d(0, 0, 0));
	LandmarkDetector::DetectLandmarksInImageUseInitOri(img, faceBox, landmark_model, landmark_param, rotation_hypothese);
	printf("init over\n");


	initShapes(landmark_model);

	return 1;
}

double  Landmark_clnf::get_5points_dis(vector<cv::Point>& landmark1, vector<cv::Point>& landmark2)
{
	double  total_dist = 0;
	for (int i = 0; i < landmark1.size(); i++)
	{
		total_dist += sqrtf((landmark1[i].x - landmark2[i].x) * (landmark1[i].x - landmark2[i].x)
			+ (landmark1[i].y - landmark2[i].y) * (landmark1[i].y - landmark2[i].y));
	}
	return total_dist;
}

void  Landmark_clnf::conv_landmark68_2_landmark5(vector<cv::Point>& landmark5, vector<cv::Point>& landmark68)
{
	landmark5.clear();
	//左眼
	cv::Point  left_eye68 = cv::Point(0, 0);
	for (int i = 36; i < 42; i++)
	{
		left_eye68.x += landmark68[i].x;
		left_eye68.y += landmark68[i].y;
	}
	left_eye68.x /= 6;
	left_eye68.y /= 6;
	landmark5.push_back(left_eye68);

	//右眼
	cv::Point  right_eye68 = cv::Point(0, 0);
	for (int i = 42; i < 48; i++)
	{
		right_eye68.x += landmark68[i].x;
		right_eye68.y += landmark68[i].y;
	}
	right_eye68.x /= 6;
	right_eye68.y /= 6;
	landmark5.push_back(right_eye68);

	landmark5.push_back(landmark68[30]);  //nose
	landmark5.push_back(landmark68[48]);  //mouth_left
	landmark5.push_back(landmark68[54]);  //mouth_right
}

void  Landmark_clnf::use_pdm_get_initShape(cv::Rect& face, LandmarkDetector::CLNF& landmark_model, cv::Vec3d& rotation_hypothese, vector<cv::Point>& landmark)
{
	landmark.clear();

	// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
	landmark_model.pdm.CalcParams(landmark_model.params_global, face, landmark_model.params_local, rotation_hypothese);

	// Placeholder for the landmarks
	cv::Mat_<float> current_shape(2 * landmark_model.pdm.NumberOfPoints(), 1, 0.0f);
	landmark_model.pdm.CalcShape2D(current_shape, landmark_model.params_local, landmark_model.params_global);
	int ld_n = current_shape.rows / 2;
	for (int i = 0; i < ld_n; ++i)
	{
		cv::Point featurePoint(cvRound(current_shape.at<float>(i)), cvRound(current_shape.at<float>(i + ld_n)));
		landmark.push_back(featurePoint);
	}
}

void  Landmark_clnf::generateEuler(vector<cv::Vec3d>& rotation_hypothese)
{
	rotation_hypothese.clear();
	float  pitch_degree = 0;
	float  yaw_degree = 0;
	float  roll_degree = 0;
	for (int pitch = -30; pitch <= 30; pitch += 5)
	{
		pitch_degree = pitch * CVV_PI / 180.0;
		for (int yaw = -90; yaw <= 90; yaw += 5)
		{
			yaw_degree = yaw * CVV_PI / 180.0;
			for (int roll = -50; roll <= 50; roll += 5)
			{
				roll_degree = roll * CVV_PI / 180.0;
				rotation_hypothese.push_back(cv::Vec3d(pitch_degree, yaw_degree, roll_degree));
			}
		}
	}
}

//保证两只眼睛之间的距离固定，重新计算model的5个点
void  Landmark_clnf::refineModelLdmark(vector<cv::Point>& landmark5)
{
	int  m_cx = FACE_RECT_X + FACE_RECT_W / 2;
	int  m_cy = FACE_RECT_Y + FACE_RECT_H / 2;

	float  dist = sqrtf((landmark5[0].x - landmark5[1].x)*(landmark5[0].x - landmark5[1].x)
		+ (landmark5[0].y - landmark5[1].y)*(landmark5[0].y - landmark5[1].y));

	float  scale = dist / FACE_EYE_DIST;

	for (int i = 0; i < landmark5.size(); i++)
	{
		int off_x = landmark5[i].x - m_cx;
		int off_y = landmark5[i].y - m_cy;
		landmark5[i].x = m_cx + off_x / scale;
		landmark5[i].y = m_cy + off_y / scale;
	}
}

void  Landmark_clnf::initShapes(LandmarkDetector::CLNF& landmark_model)
{
	cv::Rect face = cv::Rect(FACE_RECT_X, FACE_RECT_Y, FACE_RECT_W, FACE_RECT_H);
	m_shapeCaches.clear();
	for (int i = 0; i < rotation_hypotheses_inits.size(); i++)
	{
		//计算所有形状
		vector<cv::Point> landmark;
		use_pdm_get_initShape(face, landmark_model, rotation_hypotheses_inits[i], landmark);

		//提取每个形状的5个点
		shapeCaches  shape;
		shape.rotation_hypothese = rotation_hypotheses_inits[i];		
		conv_landmark68_2_landmark5(shape.landmark5, landmark);

		//重新矫正模型的5个点，保证两个眼睛距离固定
		refineModelLdmark(shape.landmark5);

		m_shapeCaches.push_back(shape);
	}
}

//获取当前人脸的ldmark的变化量
void  Landmark_clnf::getCurLdmarkCof(cv::Rect& face, vector<cv::Point>& landmark5, float& offset_x, float& offset_y, float& scale)
{
	int  m_cx = FACE_RECT_X + FACE_RECT_W / 2;
	int  m_cy = FACE_RECT_Y + FACE_RECT_H / 2;
	int  c_cx = face.x + face.width / 2;
	int  c_cy = face.y + face.height / 2;
	offset_x = c_cx - m_cx;
	offset_y = c_cy - m_cy;

	float  dist = sqrtf((landmark5[0].x - landmark5[1].x)*(landmark5[0].x - landmark5[1].x)
		+ (landmark5[0].y - landmark5[1].y)*(landmark5[0].y - landmark5[1].y));

	scale = dist / FACE_EYE_DIST;
}

//矫正当前人脸的ldmark位置与model对齐
void Landmark_clnf::refineLdmark(cv::Mat& img, cv::Rect& face, vector<cv::Point>& landmark5)
{
	//获取当前人脸的ldmark的变化量
	float offset_x, offset_y, scale;
	getCurLdmarkCof(face, landmark5, offset_x, offset_y, scale);

	//矫正当前人脸的ldmark位置与model对齐
	int  m_cx = FACE_RECT_X + FACE_RECT_W / 2;
	int  m_cy = FACE_RECT_Y + FACE_RECT_H / 2;
	int  c_cx = face.x + face.width / 2;
	int  c_cy = face.y + face.height / 2;
	for (int i = 0; i < landmark5.size(); i++)
	{
		landmark5[i].x = m_cx + (landmark5[i].x - c_cx) / scale;
		landmark5[i].y = m_cy + (landmark5[i].y - c_cy) / scale;
	}
}

void  Landmark_clnf::clc_shape_pdm(cv::Mat& img, cv::Rect& face, vector<cv::Point>& landmark5,  int faceDetectType, vector<cv::Vec3d>& rotation_hypothese)
{
	rotation_hypothese.clear();	
	//如果不是mtcnn检测的人脸
	if (faceDetectType == 2)
	{
		rotation_hypothese.push_back(cv::Vec3d(0, 0, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, -0.5236, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, 0.5236, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, -0.96, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, 0.96, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, 0, 0.5236));
		rotation_hypothese.push_back(cv::Vec3d(0, 0, -0.5236));
		rotation_hypothese.push_back(cv::Vec3d(0, -1.57, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, 1.57, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, -1.22, 0.698));
		rotation_hypothese.push_back(cv::Vec3d(0, 1.22, -0.698));
		return;
	}

	//矫正当前人脸的ldmark位置与model对齐
	vector<cv::Point> refine_landmark5(landmark5);
	refineLdmark(img, face, refine_landmark5);

	//充模型库中匹配一个距离最近的模型作为初始shape
	vector<float> dist_caches;
	int id = 0;
	float  min_dis = 100000;
	for (int i = 0; i < m_shapeCaches.size(); i++)
	{
		double dis = get_5points_dis(refine_landmark5, m_shapeCaches[i].landmark5);
		dist_caches.push_back(dis);
		if (dis < min_dis)
		{
			min_dis = dis;
			id = i;
		}
	}
	rotation_hypothese.push_back(m_shapeCaches[id].rotation_hypothese);
}

//// Correct the box to expectation to be tight around facial landmarks  FaceDetectMTCNN.cpp ---887
void  Landmark_clnf::refineMtcnnBox(cv::Rect& box, vector<cv::Point>& landmarks, int faceDetectType)
{
	//如果不是mtcnn检测的人脸
	if (faceDetectType == 2)
	{
		return;
	}

	box.x = box.width * -0.0075 + box.x;
	box.y = box.height * 0.2459 + box.y;
	box.width = 1.0323 * box.width;
	box.height = 0.7751 * box.height;
}

void Landmark_clnf::getInitShape(cv::Mat& img, cv::Rect& mtcnn_faceBox, vector<cv::Point>& mtcnn_ldmark, vector<cv::Point>& init_landmark)
{
	int  faceDetectType = 1;
	if (mtcnn_ldmark.size() == 0)
	{
		faceDetectType = 2;
	}
	//重新矫正faceBox
	cv::Rect  faceBox = mtcnn_faceBox;
	refineMtcnnBox(faceBox, mtcnn_ldmark, faceDetectType);

	//根据mtcnn的5个关键点，计算初始形状
	vector<cv::Vec3d> rotation_hypothese;
	clc_shape_pdm(img, faceBox, mtcnn_ldmark, faceDetectType, rotation_hypothese);

	init_landmark.clear();
	use_pdm_get_initShape(faceBox, landmark_model, rotation_hypothese[0], init_landmark);
}

static  double clnf_totalTime = 0;
static  int  clnf_totalFaceCount = 0;
vector<vector<cv::Point> >  Landmark_clnf::LandmarkImage(cv::Mat& img, vector<cv::Rect>& mtcnn_faceBoxes, vector<vector<cv::Point> >& mtcnn_landmarks)
{
	vector<vector<cv::Point> > face_landmark68;
	if (img.empty())
		return face_landmark68;

	//如果没有landmark，则直接用原始clnf预测11次
	int  faceDetectType = 1;
	if (mtcnn_landmarks.size() == 0)
	{
		faceDetectType = 2;
	}
	
	for (unsigned int i = 0; i < mtcnn_faceBoxes.size(); i++) 
	{		
		//测试时间
		double start = cvGetTickCount() / (1000.0 * cvGetTickFrequency());

		//重新矫正faceBox
		cv::Rect  faceBox = mtcnn_faceBoxes[i];		
		refineMtcnnBox(faceBox, mtcnn_landmarks[i], faceDetectType);

		//根据mtcnn的5个关键点，计算初始形状
		vector<cv::Vec3d> rotation_hypothese;
		clc_shape_pdm(img, faceBox, mtcnn_landmarks[i], faceDetectType, rotation_hypothese);

		double start1 = cvGetTickCount() / (1000.0 * cvGetTickFrequency());
		double cousumeTime1 = start1 - start;

		// landmark		
		bool success = LandmarkDetector::DetectLandmarksInImageUseInitOri(img, faceBox, landmark_model, landmark_param, rotation_hypothese);
		
		vector<cv::Point>  landmark68;
		int n = landmark_model.detected_landmarks.rows / 2;
		for (int l = 0; l < n; l++) {
			cv::Point featurePoint(cvRound(landmark_model.detected_landmarks.at<float>(l)), cvRound(landmark_model.detected_landmarks.at<float>(l + n)));
			landmark68.push_back(featurePoint);
		}
		face_landmark68.push_back(landmark68);

		double end = cvGetTickCount() / (1000.0 * cvGetTickFrequency());
		double cousumeTime = end - start;
		clnf_totalTime += cousumeTime;
		clnf_totalFaceCount += 1;
		//printf("clnf fit one face: %f, %f\n", cousumeTime, cousumeTime1);
	}

	//printf("clnf: %d, %f, %f\n\n", clnf_totalFaceCount, clnf_totalTime, clnf_totalTime / clnf_totalFaceCount);

	return face_landmark68;
}