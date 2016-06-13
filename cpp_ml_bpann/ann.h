#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <random>
#include <functional>
#include <initializer_list>
#include <memory>
using namespace std;
/*管理矩阵类内存*/
template<class _Ty>
class Pointer
{
public:
	Pointer();
	~Pointer();
	_Ty* _base;
};
template<class _Ty>
inline Pointer<_Ty>::Pointer()
{
	_base = nullptr;
}

template<class _Ty>
inline Pointer<_Ty>::~Pointer()
{
	if (_base == nullptr)
		return;
	delete[] _base;
}


/*存储数据的矩阵*/
template<class _Ty>
class Mat
{
public:
	Mat();
	Mat(size_t rows, size_t cols);
	Mat(initializer_list<_Ty> list);
	~Mat();
	Mat<_Ty>&  operator=(const Mat<_Ty> &mat);
	_Ty& at(size_t x, size_t y = 0);
	_Ty& operator[](int);
	_Ty* begin();
	_Ty* end();
	void fill(const _Ty &val);
	size_t rows;
	size_t cols;
	shared_ptr<Pointer<_Ty>> ptr = make_shared<Pointer<_Ty>>();
};


template<class _Ty>
inline Mat<_Ty>::Mat() :Mat(0, 0)
{
}

template<class _Ty>
inline Mat<_Ty>::Mat(size_t rows, size_t cols)
{
	this->rows = rows;
	this->cols = cols;
	ptr->_base = new _Ty[rows*cols];
	
}
template<class _Ty>
inline Mat<_Ty>::Mat(initializer_list<_Ty> lst)
{
	this->cols = lst.size();
	this->rows = 1;
	ptr->_base = new _Ty[cols];
	for (size_t i = 0; i < lst.size(); i++)
		*(ptr->_base + i) = *(lst.begin() + i);
}
template<class _Ty>
inline Mat<_Ty>::~Mat()
{
}
template<class _Ty>
inline Mat<_Ty>& Mat<_Ty>::operator=(const Mat<_Ty>& mat)
{
	if (this == &mat)
		return *this;
	this->cols = mat.cols;
	this->rows = mat.rows;
	this->ptr = mat.ptr;
	return *this;
}
template<class _Ty>
inline _Ty & Mat<_Ty>::at(size_t x, size_t y)
{
	return *(ptr->_base + y*cols + x);
}
template<class _Ty>
inline _Ty & Mat<_Ty>::operator[](int i)
{
	return this->at(i);
}
template<class _Ty>
inline _Ty * Mat<_Ty>::begin()
{
	return this->ptr->_base;
}
template<class _Ty>
inline _Ty * Mat<_Ty>::end()
{
	return this->ptr->_base + rows*cols;
}
template<class _Ty>
inline void Mat<_Ty>::fill(const _Ty &val)
{
	std::fill_n(ptr->_base, rows*cols, val);
}
// 矩阵类结束

// 网络类开始
class ANN
{
public:
	ANN();
	~ANN();
	void SetLayers(Mat<int> layers);
	void SetTrainData(Mat<double> samples, Mat<double> responses);
	void SetStudyRate(double scale = 0.1);
	void SetThreshold(double threshold = 1.0);
	void SetTermIterations(int iterations = 5000);
	void SetTermErrorRate(double error = 0.01);
	void Train();
	void Predict(Mat<double>& sample,Mat<double>& response);
private:
	void init_weights();
	void forward();
	void backward();
	
	Mat<double> samples;
	Mat<double> responses;
	vector<Mat<double>> weights; // 容器每元素为一层，矩阵行为单元，列是对应权值
	Mat<double> outputs;
	vector<Mat<double>> delta_weights;

	double theta;
	double eta;
	double max_iter;
	double max_error;

};


