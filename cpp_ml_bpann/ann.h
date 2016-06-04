#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <random>
#include <functional>
#include <initializer_list>
using namespace std;
/*
 
 */
 // ����
template<class _Ty>
class Mat
{
public:
	Mat();
	Mat(size_t row, size_t col);
	Mat(initializer_list<_Ty> list);
	~Mat();
	_Ty& at(size_t x, size_t y = 0);
	_Ty& operator[](int);
	_Ty* begin();
	_Ty* end();
	void fill(const _Ty &val);
	size_t row;
	size_t col;
	
private:
	_Ty* _db;

};

template<class _Ty>
inline Mat<_Ty>::Mat() :Mat(0, 0)
{
}

template<class _Ty>
inline Mat<_Ty>::Mat(size_t row, size_t col)
{
	this->row = row;
	this->col = col;
	_db = new _Ty[row*col];
}
template<class _Ty>
inline Mat<_Ty>::Mat(initializer_list<_Ty> lst)
{
	this->col = lst.size();
	this->row = 1;
	_db = new _Ty[col];
	for (size_t i = 0; i < lst.size(); i++)
		*(_db + i) = *(lst.begin() + i);
}
template<class _Ty>
inline Mat<_Ty>::~Mat()
{
	delete _db;
}
template<class _Ty>
inline _Ty & Mat<_Ty>::at(size_t x, size_t y)
{
	return *(_db + y*col + x);
}
template<class _Ty>
inline _Ty & Mat<_Ty>::operator[](int i)
{
	return this->at(i);
}
template<class _Ty>
inline _Ty * Mat<_Ty>::begin()
{
	return this->_db;
}
template<class _Ty>
inline _Ty * Mat<_Ty>::end()
{
	return this->_db + row*col;
}
template<class _Ty>
inline void Mat<_Ty>::fill(const _Ty &val)
{
	std::fill_n(_db, row*col, val);
}
// ���������

// �����࿪ʼ
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
	void Perdict(Mat<double> sample,Mat<double> response);
private:
	void init_weights();
	void forward();
	void backward();
	
	Mat<double> samples;
	Mat<double> responses;
	vector<Mat<double>> weights; // ����ÿԪ��Ϊһ�㣬������Ϊ��Ԫ�����Ƕ�ӦȨֵ
	Mat<double> outputs;
	vector<Mat<double>> delta_weights;

	double theta;
	double eta;

};