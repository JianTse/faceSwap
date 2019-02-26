#include "swap.h"

faceSwap::faceSwap()
{
}
faceSwap::~faceSwap()
{
}

void faceSwap::delaunayTriangulation(const std::vector<Point2f>& hull, std::vector<correspondens>& delaunayTri, Rect rect)
{
	cv::Subdiv2D subdiv(rect);
	for (int it = 0; it < hull.size(); it++)
		subdiv.insert(hull[it]);
	//cout<<"done subdiv add......"<<endl;
	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	//cout<<"traingleList number is "<<triangleList.size()<<endl;



	//std::vector<Point2f> pt;
	//correspondens ind;
	for (size_t i = 0; i < triangleList.size(); ++i)
	{

		std::vector<Point2f> pt;
		correspondens ind;
		Vec6f t = triangleList[i];
		pt.push_back(Point2f(t[0], t[1]));
		pt.push_back(Point2f(t[2], t[3]));
		pt.push_back(Point2f(t[4], t[5]));
		//cout<<"pt.size() is "<<pt.size()<<endl;

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			//cout<<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<t[3]<<" "<<t[4]<<" "<<t[5]<<endl;
			int count = 0;
			for (int j = 0; j < 3; ++j)
				for (size_t k = 0; k < hull.size(); k++)
					if (abs(pt[j].x - hull[k].x) < 1.0   &&  abs(pt[j].y - hull[k].y) < 1.0)
					{
						ind.index.push_back(k);
						count++;
					}
			if (count == 3)
				//cout<<"index is "<<ind.index[0]<<" "<<ind.index[1]<<" "<<ind.index[2]<<endl;
				delaunayTri.push_back(ind);
		}
		//pt.resize(0);
		//cout<<"delaunayTri.size is "<<delaunayTri.size()<<endl;
	}


}


void faceSwap::warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &t1, std::vector<Point2f> &t2)
{

	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	// Offset points by left top corner of the respective rectangles
	std::vector<Point2f> t1Rect, t2Rect;
	std::vector<Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y)); // for fillConvexPoly

	}

	// Get mask by filling triangle
	Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

	// Apply warpImage to small rectangular patches
	Mat img1Rect;
	img1(r1).copyTo(img1Rect);

	Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

	multiply(img2Rect, mask, img2Rect);
	multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	img2(r2) = img2(r2) + img2Rect;

}

void faceSwap::applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
{
	// Given a pair of triangles, find the affine transform.
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	// Apply the Affine Transform just found to the src image
	warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, BORDER_REFLECT_101);
}

void faceSwap::merge(cv::Mat& img1, cv::Mat& img2, vector<cv::Point>& landmarks68, cv::Mat& dst)
{
	//calculate mask
	std::vector<Point> hull8U;
	for (int i = 0; i< 17; ++i)
	{
		Point pt(landmarks68[i].x, landmarks68[i].y);
		hull8U.push_back(pt);
	}
	for (int i = 26; i >= 17; i--)
	{
		Point pt(landmarks68[i].x, landmarks68[i].y);
		hull8U.push_back(pt);
	}
	Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255, 255, 255));	

	Rect r = boundingRect(hull8U);
	Point center = cv::Point(r.x + r.width / 2, r.y + r.height / 2);
	Mat output;
	cv::seamlessClone(img1, img2, mask, center, output, cv::NORMAL_CLONE);
}

void  drawTriangulation(cv::Mat& img, std::vector<Point2f>& pts)
{
	cv::Point p1, p2, p3;
	p1 = cv::Point(int(pts[0].x), int(pts[0].y));
	p2 = cv::Point(int(pts[1].x), int(pts[1].y));
	p3 = cv::Point(int(pts[2].x), int(pts[2].y));
	cv::line(img, p1,p2, cv::Scalar(255,0,0));
	cv::line(img, p2, p3, cv::Scalar(255, 0, 0));
	cv::line(img, p3, p1, cv::Scalar(255, 0, 0));
}

int faceSwap::process(cv::Mat& img1, vector<cv::Point>& landmark1, cv::Mat& img2, vector<cv::Point>& landmark2, cv::Mat& dst)
{
	cv::Mat imgCV1 = img1.clone();
	cv::Mat imgCV2 = img2.clone();

	//---------------------step 3. find convex hull -------------------------------------------
	Mat imgCV1Warped = imgCV2.clone();
	imgCV1.convertTo(imgCV1, CV_32F);
	imgCV1Warped.convertTo(imgCV1Warped, CV_32F);

	std::vector<Point2f> hull1;
	std::vector<Point2f> hull2;
	std::vector<int> hullIndex;

	cv::convexHull(landmark2, hullIndex, false, false);

	for (int i = 0; i < hullIndex.size(); i++)
	{
		hull1.push_back(landmark1[hullIndex[i]]);
		hull2.push_back(landmark2[hullIndex[i]]);
	}


	//-----------------------step 4. delaunay triangulation -------------------------------------	
	std::vector<correspondens> delaunayTri;
	Rect rect(0, 0, imgCV1Warped.cols, imgCV1Warped.rows);
	delaunayTriangulation(hull2, delaunayTri, rect);

	for (size_t i = 0; i<delaunayTri.size(); ++i)
	{
		std::vector<Point2f> t1, t2;
		correspondens corpd = delaunayTri[i];
		for (size_t j = 0; j<3; ++j)
		{
			t1.push_back(hull1[corpd.index[j]]);
			t2.push_back(hull2[corpd.index[j]]);
		}

		//cv::Mat img = img1.clone();
		//drawTriangulation(img, t1);
		//cv::imshow("img", img);
		//cv::waitKey(0);

		warpTriangle(imgCV1, imgCV1Warped, t1, t2);
	}

	//------------------------step 5. clone seamlessly -----------------------------------------------

	//calculate mask
	std::vector<Point> hull8U;

	for (int i = 0; i< hull2.size(); ++i)
	{
		Point pt(hull2[i].x, hull2[i].y);
		hull8U.push_back(pt);
	}


	Mat mask = Mat::zeros(imgCV2.rows, imgCV2.cols, imgCV2.depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255, 255, 255));

	Rect r = boundingRect(hull2); 
	Point center = (r.tl() + r.br()) / 2;

	imgCV1Warped.convertTo(imgCV1Warped, CV_8UC3);
	seamlessClone(imgCV1Warped, imgCV2, mask, center, dst, NORMAL_CLONE);

	return 0;
}

