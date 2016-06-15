#include "ann.h"
enum Config
{
	FEATNUM = 4,
	CLASS_NUM = 3,
	SMP_EVERY = 50,
	SMP_TOTAL = 150
};

void split(string &str, vector<string> &strv)
{
	string::iterator bg, ed;
	bg = ed = str.begin();
	for_each(str.begin(), str.end(),
		[&bg, &ed, &strv](const auto& a)
	{
		if (*ed == ',')
		{
			strv.push_back(string(bg, ed));
			bg = ++ed;
		}
		else
		{
			++ed;
		}
	});
	strv.push_back(string(bg, str.end()));
}

void read(Mat<double>& td,Mat<double>& rs, vector<Mat<double>>&tt, vector<Mat<double>>& trs)
{
	ifstream fs;
	fs.open("iris.csv");
	vector<string> strv;
	while (!fs.eof())
	{
		string str;
		getline(fs, str);
		strv.push_back(str);
	}
	double k = 4.0 / 5.0;
	td = Mat<double>(strv.size()*k, FEATNUM);
	rs = Mat<double>(strv.size()*k, CLASS_NUM);
	rs.fill(0);
	int idx = 0;
	for (int c = 0; c < CLASS_NUM; c++)
	{
		for (int i = SMP_EVERY*c; i < SMP_EVERY*(c + k); i++)
		{
			vector<string> tmp;
			split(strv[i], tmp);
			int j = 0;
			for (; j < FEATNUM; j++)
			{
				td.at(j, idx) = atof(tmp[j].c_str());
			}
			double r = atof(tmp[j].c_str());
			rs.at(r, idx) = 1;
			idx++;
		}
	}
	/// 读入一些测试数据
	for (int c = 0; c < CLASS_NUM; c++)
	{
		for (int i = SMP_EVERY*(c + k); i < SMP_EVERY*(c + 1); i++)
		{
			vector<string> tmp;
			split(strv[i], tmp);
			int j = 0;
			Mat<double> tm(1, FEATNUM);
			Mat<double> rm(1, CLASS_NUM);
			for (; j < FEATNUM; j++)
			{
				tm[j] = atof(tmp[j].c_str());
			}
			double r = atof(tmp[j].c_str());
			rm.fill(0);
			rm[r] = 1;
			tt.push_back(tm);
			trs.push_back(rm);
		}
	}

}



int main()
{
	shared_ptr<ANN> Network = make_shared<ANN>();
	Network->SetLayers(Mat<int>{4,12,12, 3});
	Mat<double> traindata;
	Mat<double>	result; 
	vector<Mat<double>> test;
	vector <Mat<double>> test_results;
	read(traindata, result, test,test_results);
	Network->SetTrainData(traindata, result);
	Network->Train();
	int success = 0;
	function<int(Mat<double>)> max_idx = 
		[](Mat<double> &mat) -> int 
	{
		double max = FLT_MIN;
		double maxidx = 0;
		for (int i = 0; i < mat.cols; i++)
		{
			if (mat[i] > max)
			{
				maxidx = i;
				max = mat[i];
			}
		}
		return maxidx;
	};
	for (int i = 0; i < test.size(); i++)
	{
		Mat<double> td = test[i];
		Mat<double> rlt;
		Network->Predict(td, rlt);
		cout << "样本 " << i+1 << endl;
		cout << "预测结果" << rlt <<  "\t真实结果" << test_results[i] << endl;
		if (max_idx(rlt) == max_idx(test_results[i]))
		{
			success++;
			cout << "预测成功！" << endl;
		}
		else
		{
			cout << "预测失败。" << endl;
		}
		system("pause");
	}
	cout << "成功率:" << success * 100.0/test.size() << "%"  << endl;
	system("pause");
}

ANN::ANN()
{
	theta = 1.0;
	eta = 0.1;
	max_iter = 100000;
	max_error = 0.000001;
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
	outputs.fill(0);
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
	normalize();
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
			backward(i);
			for (int j = 0; j < weights[weights.size() - 1].rows; j++)
			{
				error += pow(outputs.at(j, weights.size() - 1) - responses.at(j, i), 2);
			}

		}
		error /= 2;
		if (error < max_error)
		{
			cout << t << endl;
			
			return;
		}
		/*
		if (t % 5000 == 0)
		{
			cout << "迭代次数:" << t << "\t错误率:" << error << endl;
		}*/
	}
}

void ANN::Predict(Mat<double>& sample, Mat<double>& response)
{
	assert(sample.cols <= outputs.cols);
	for (size_t j = 0; j < sample.cols; j++)
	{
		if (min_and_max.at(j, 0) == min_and_max.at(j, 1) || sample[j] > min_and_max.at(j, 1))
		{
			outputs[j] = 1;
		}
		else if (sample[j] < min_and_max.at(j, 0))
		{
			outputs[j] = 0;
		}
		else
		{
			outputs[j] = sample[j] / (min_and_max.at(j, 1) - min_and_max.at(j, 0));
		}
		
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
			tmp -= theta;
			tmp = 1 / (1 + exp(-tmp));
			outputs.at(j, i) = tmp;
		}
	}
}

void ANN::backward(int &idx)
{
	// 输出层反向传播
	int output_layer = outputs.rows - 1;
	for (int j = 0; j < weights[output_layer].rows; j++)
	{
		//double delta_weight = (response[i] - layers[outputIdx][i].Result) * layers[outputIdx][i].Result * (1 - layers[outputIdx][i].Result);
		double &result_j = outputs.at(j, output_layer);
		double delta_weight = (responses.at(j,idx) - result_j) * result_j *(1 - outputs.at(j, output_layer));
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
				sum += delta_weights[i + 1].at(j, k) * weights[i + 1].at(j, k); //下游所有有关权值更新
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

void ANN::normalize()
{
	min_and_max = Mat<double>(2, samples.cols);
	for (int i = 0; i < samples.cols; i++)
	{
		min_and_max.at(i, 0) = FLT_MAX;
		min_and_max.at(i, 1) = FLT_MIN;
	}
	// 求极值
	for (int i = 0; i < samples.rows; i++)
	{
		for (int j = 0; j < samples.cols; j++)
		{
			if (min_and_max.at(j, 0) > samples.at(j, i))
				min_and_max.at(j, 0) = samples.at(j, i);
			if (min_and_max.at(j, 1) < samples.at(j, i))
				min_and_max.at(j, 1) = samples.at(j, i);

		}
	}
	for (int i = 0; i < samples.rows; i++)
	{
		for (int j = 0; j < samples.cols; j++)
		{
			if (min_and_max.at(j, 1) == min_and_max.at(j, 0))
			{
				samples.at(j, i) = 1;
			}
			else
			{
				samples.at(j, i) = samples.at(j, i) / (min_and_max.at(j, 1) - min_and_max.at(j, 0));
			}
		}
	}
}



