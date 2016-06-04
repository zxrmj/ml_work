#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
/*
 
 */
 // 矩阵
template<class _Ty>
class Mat
{
public:
	Mat() = delete;
	Mat(size_t row, size_t col);
	~Mat();
	_Ty& at(size_t x, size_t y = 0);
	void fill(const _Ty &val);
	size_t row;
	size_t col;
private:
	_Ty* _db;

};

template<class _Ty>
inline Mat<_Ty>::Mat(size_t row, size_t col)
{
	this->row = row;
	this->col = col;
	_db = new _Ty[row*col];
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
inline void Mat<_Ty>::fill(const _Ty &val)
{
	std::fill_n(_db, row*col, val);
}
// 矩阵类结束

// 网络类开始
class ANN
{
public:
	ANN();
	~ANN();
	void train();
	void predict();
private:
	void init_weights();
	void forward();
	void backward();
	

	vector<Mat<double>> weights; // 容器每元素为一层，矩阵行为单元，列是对应权值
	Mat<double> outputs;
	vector<Mat<double>> delta_weights;

};