#ifndef _FACE_SWAP__H__
#define _FACE_SWAP__H__


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

using namespace std;
using namespace cv;
using std::string;

/*
需要opencv3.0以上版本，因为函数：seamlessClone
*/

struct correspondens{
	std::vector<int> index;
};

class faceSwap {
public:
	faceSwap();
	~faceSwap();	
	int process(cv::Mat& img1, vector<cv::Point>& landmark1, cv::Mat& img2, vector<cv::Point>& landmark2, cv::Mat& dst);

private:
	void delaunayTriangulation(const std::vector<Point2f>& hull, std::vector<correspondens>& delaunayTri, Rect rect);
	void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri);
	void warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &t1, std::vector<Point2f> &t2);
	void merge(cv::Mat& img1, cv::Mat& img2, vector<cv::Point>& landmarks, cv::Mat& dst);
};

#endif