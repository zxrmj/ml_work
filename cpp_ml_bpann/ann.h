#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <random>
#include <functional>
#include <initializer_list>
#include <thread>
#include <memory>
#include <regex>
#include <io.h>
#include <numeric>
#include <Windows.h>
#include <boost\property_tree\ptree.hpp>
#include <boost\property_tree\xml_parser.hpp>
#include <boost\progress.hpp>
#include <boost\algorithm\string.hpp>
using namespace boost::property_tree;
using namespace std;

/* 管理矩阵类数据内存 */
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
	if (_base != nullptr)
		delete[] _base;
}


/* 矩阵类，用于存储数据 */
template<class _Ty>
class Mat
{
public:
	Mat();
	Mat(size_t rows, size_t cols);
	Mat(initializer_list<_Ty> list);
	~Mat();
	Mat<_Ty>&  operator=(const Mat<_Ty> &mat);
	bool operator==(Mat<_Ty> &mat);
	bool operator!=(Mat<_Ty> &mat);
	_Ty& at(size_t x, size_t y = 0);
	_Ty& operator[](int);
	_Ty* begin();
	_Ty* end();
	void fill(const _Ty &val);
	size_t rows;
	size_t cols;
	shared_ptr<Pointer<_Ty>> ptr = make_shared<Pointer<_Ty>>();
};

template<typename _Tn>
inline ostream& operator << (ostream &os, Mat<_Tn> &m)
{
	os << "[";
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			os << m.at(j, i) << ", ";
		}
		if(i < m.rows-1)
			os << endl;
	}
	os << "]";
	return os;
}

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
inline bool Mat<_Ty>::operator==(Mat<_Ty>& mat)
{
	if (rows != mat.rows || cols != mat.cols)
		return false;
	else
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				if (this->at(j, i) != mat.at(j, i))
					return false;
			}
		}
		return true;
	}
}

template<class _Ty>
inline bool Mat<_Ty>::operator!=(Mat<_Ty>& mat)
{
	return !(this == mat);
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
	~ANN() = default;
	void SetLayers(Mat<int> layers);
	void SetLayers(initializer_list<int> init_list);
	void SetTrainData(Mat<double> samples, Mat<double> responses);
	void SetStudyRate(double scale = 0.1);
	void SetThreshold(double threshold = 1.0);
	void SetTermIterations(int iterations = 5000);
	void SetTermErrorRate(double error = 0.01);
	double GetStudyRate();
	double GetThreshold();
	double GetTermIterations();
	double GetTermErrorRate();
	void Train();
	void Predict(Mat<double>& sample,Mat<double>& response);
	void Save(string path);
	void Load(string path);
	string ToString();
	size_t GetHashCode();
private:
	void init_weights();
	void forward();
	void backward(int &idx);
	void normalize();
	void create_net(Mat<int> layers, bool init_weight);

	Mat<double> samples;
	Mat<double> responses;
	Mat<double> min_and_max;
	vector<Mat<double>> weights; // 容器每元素为一层，矩阵行为单元，列是对应权值
	Mat<double> outputs;
	vector<Mat<double>> delta_weights;

	bool trained;
	double theta;
	double eta;
	double max_iter;
	double max_error;

};


