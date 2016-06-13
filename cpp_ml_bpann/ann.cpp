#include "ann.h"

int main()
{
	shared_ptr<ANN> Network = make_shared<ANN>();
	Network->SetLayers(Mat<int>{4, 6, 6, 3});
	Mat<double> traindata({ 1,2,3,4 });
	Mat<double>	result({ 0,1,0 });
	Mat<double> test;
	Network->SetTrainData(traindata, result);
	Network->Train();
	Network->Predict(traindata, test);
	for (int i = 0; i < test.cols; i++)
	{
		cout << test[i] << endl;
	}
	system("pause");
}

ANN::ANN()
{
	theta = 1.0;
	eta = 0.1;
	max_iter = 5000;
	max_error = 0.0001;
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
		Mat<double> dw(layers[l], layers[l - 1]); // 每行一个单元所有更新权值
		weights.push_back(w);
		delta_weights.push_back(dw);
	}

	init_weights();
}

void ANN::SetTrainData(Mat<double> samples, Mat<double> responses)
{
	assert(samples.rows == responses.rows);
	this->samples = samples;
	this->responses = responses;
}

void ANN::SetStudyRate(double scale)
{
	this->eta = scale;
}

void ANN::SetThreshold(double threshold)
{
	this->theta = threshold;
}

void ANN::SetTermIterations(int iterations)
{
	this->max_iter = iterations;
}

void ANN::SetTermErrorRate(double error)
{
	this->max_error = error;
}

void ANN::Train()
{
	for (int t = 0; t < max_iter; t++)
	{
		double error = 0.0;
		for (int i = 0; i < samples.rows; i++)
		{
			for (int j = 0; j < samples.cols; j++)
			{
				outputs.at(j, 0) = samples.at(j, i);
			}
			forward();
			backward();
			for (int j = 0; j < weights[weights.size() - 1].rows; j++)
			{
				error += pow(outputs.at(j, weights.size() - 1) - responses[j], 2);
			}
		}
		error /= 2;
		if (error < max_error)
		{
			return;
		}
	}
}

void ANN::Predict(Mat<double>& sample, Mat<double>& response)
{
	assert(sample.cols <= outputs.cols);
	for (size_t j = 0; j < sample.cols; j++)
	{
		outputs[j] = sample[j];
	}
	forward();
	int output_layer = outputs.rows - 1;
	response = Mat<double>(1, weights[output_layer].rows);
	for (size_t j = 0; j < weights[output_layer].rows; j++)
	{
		response[j] = outputs.at(j,output_layer);
	}
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

void ANN::forward()
{
	for (int i = 1; i < outputs.rows; i++)
	{
		for (int j = 0; j < weights[i].rows; j++)
		{
			// weights * inputs;
			double tmp = 0;
			for (int k = 0; k < weights[i].cols; k++)
			{
				tmp += weights[i].at(k, j) * outputs.at(k, i - 1);
			}
			outputs.at(j, i) = tmp;
		}
	}
}

void ANN::backward()
{
	// 输出层反向传播
	int output_layer = outputs.rows - 1;
	for (int j = 0; j < weights[output_layer].rows; j++)
	{
		//double delta_weight = (response[i] - layers[outputIdx][i].Result) * layers[outputIdx][i].Result * (1 - layers[outputIdx][i].Result);
		double &result_j = outputs.at(j, output_layer);
		double delta_weight = (responses[j] - result_j) * result_j *(1 - outputs.at(j, output_layer));
		for (int k = 0; k < weights[output_layer].cols; k++)
		{
			double d = eta * delta_weight * outputs.at(k, output_layer - 1);
			weights[output_layer].at(k, j) += d;
			delta_weights[output_layer].at(k, j) = d;
		}
	}
	// 隐藏层反向传播
	for (int i = output_layer - 1; i > 0; i--)
	{
		//本层
		for (int j = 0; j < weights[i].rows; j++)
		{
			double sum = .0;
			// 下游
			for (int k = 0; k < weights[i + 1].rows; k++)
			{
				sum += delta_weights[i + 1].at(j, k); //下游所有有关权值更新
			}
			double &result = outputs.at(j, i);
			double delta_weight = (1 - result)* result*sum;
			for (int k = 0; k < weights[i].cols; k++)
			{
				double d = eta * delta_weight * outputs.at(k, i - 1);
				weights[i].at(k, j) += d;
				delta_weights[i].at(k, j) = d;
			}
		}
	}
}
