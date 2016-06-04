#include "ann.h"

int main()
{
	/*Mat<int> mat = Mat<int>({ 2,3,4,6,1 });
	for (size_t i = 0; i < mat.rows; i++)
	{
		for (size_t j = 0; j < mat.cols; j++)
		{
			cout << mat.at(j, i) << endl;
		}
	}*/
	shared_ptr<ANN> Network = make_shared<ANN>();
	Network->SetLayers(Mat<int>{4, 3, 3, 8});
	system("pause");
}

ANN::ANN()
{
	theta = 1.0;
	eta = 0.1;
}

ANN::~ANN()
{
	
}

void ANN::SetLayers(Mat<int> layers)
{
	assert(layers.rows == 1 && layers.cols > 2);
	function<int(void)> findmax = [&layers]() -> int {
		int max = INT_MIN;
		for (size_t i = 0; i < layers.cols; i++)
		{
			if (layers[i] > max)
				max = layers[i];
		}
		return max;
	};
	outputs = Mat<double>(layers.cols, findmax());
	weights.clear();
	weights.push_back(Mat<double>());
	delta_weights.clear();
	delta_weights.push_back(Mat<double>());
	for (int l = 1; l < layers.cols; l++)
	{
		Mat<double> w(layers[l], layers[l - 1]); // 每行一个单元所有权值。每列一个权值
		Mat<double> dw(layers[l], layers[l - 1]); // 每行一个单元所有更新权值。每列一个权值
		weights.push_back(w);
		delta_weights.push_back(dw);
	}

	init_weights();
}

void ANN::init_weights()
{
	mt19937 gen;
	uniform_real_distribution<> urd(0,0.1);
	auto random = bind(urd, gen);
	for (size_t l = 1; l < weights.size(); l++)
	{
		for (size_t i = 0; i < weights[l].rows; i++)
		{
			for (size_t j = 0; j < weights[l].cols; j++)
			{
				weights[l].at(j, i) = random();
			}
		}
	}
}
