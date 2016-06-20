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
	// ����һЩ��������
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
	SetConsoleTitleA(LPCSTR("����ѧϰ�γ����:���򴫲��˹�������"));
	system("color 3F");
	shared_ptr<ANN> Network = make_shared<ANN>();
	Network->SetLayers({ 4,6,6,3 });
	Network->SetStudyRate(0.2);
	Network->SetThreshold(1.5);
	Network->SetTermIterations(10000);
	Mat<double> traindata;
	Mat<double>	result; 
	vector<Mat<double>> test;
	vector <Mat<double>> test_results;
	read(traindata, result, test,test_results);
	cout << "��ȡѵ�����ɹ�" << endl;
	Network->SetTrainData(traindata, result);
	Network->Train();
	cout << Network->ToString() << endl;
	cout << "���������ʼ����" << endl;
	system("pause");
	int success = 0;
	function<int(Mat<double>)> max_idx = 
		[](Mat<double> &mat) -> int 
	{
		double max = FLT_MIN;
		int maxidx = 0;
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
		cout << "-------���� " << i+1 << "-------" << endl;
		cout << "��������" << td << endl;
		cout << "Ԥ����" << rlt <<  "\t��ʵ���" << test_results[i] << endl;
		if (max_idx(rlt) == max_idx(test_results[i]))
		{
			success++;
			cout << "-------Ԥ��ɹ���-------\n\n" << endl;
		}
		else
		{
			cout << "-------Ԥ��ʧ�ܡ�-------\n\n" << endl;
		}
	}
	cout << "�ɹ���:" << success * 100.0/test.size() << "%"  << endl;
	Network->Save("test.xml");
	shared_ptr<ANN> Network2 = make_shared<ANN>();
	Network2->Load("test.xml");
	Network2->Save("test2.xml");
	system("pause");
}

ANN::ANN()
{
	theta = 1.0;
	eta = 0.1;
	max_iter = 10000;
	max_error = 0.01;
}


void ANN::SetLayers(Mat<int> layers,bool init_weight)
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
	weights.push_back(Mat<double>(layers[0],1));
	delta_weights.clear();
	delta_weights.push_back(Mat<double>(layers[0], 1));
	for (int l = 1; l < layers.cols; l++)
	{
		Mat<double> w(layers[l], layers[l - 1]); // ÿ��һ����Ԫ����Ȩֵ��ÿ��һ��Ȩֵ
		Mat<double> dw(layers[l], layers[l - 1]); // ÿ��һ����Ԫ���и���Ȩֵ
		weights.push_back(w);
		delta_weights.push_back(dw);
	}
	if (init_weight)
	{
		init_weights();
	}
	trained = false;
}

/// <summary>
/// ����������ṹ
/// </summary>
/// <param name="layers">������Ϣ��ά��Ϊ������������ÿ��Ԫ��ֵΪ��Ӧ������Ԫ��</param>
void ANN::SetLayers(Mat<int> layers)
{
	SetLayers(layers, true);
}

void ANN::SetLayers(initializer_list<int> init_list)
{
	return SetLayers(Mat<int>(init_list));
}



/// <summary>
/// ����ѵ������
/// </summary>
/// <param name="samples">�������󣬾���ÿһ��ӦΪһ����������</param>
/// <param name="responses">�������������</param>
void ANN::SetTrainData(Mat<double> samples, Mat<double> responses)
{
	assert(samples.rows == responses.rows);
	this->samples = samples;
	this->responses = responses;
	normalize();
}

/// <summary>
/// ����ѧϰ����
/// </summary>
/// <param name="scale">ѧϰ����</param>
void ANN::SetStudyRate(double rate)
{
	this->eta = rate;
}

/// <summary>
/// ������ֵ
/// </summary>
/// <param name="threshold">��ֵ</param>
void ANN::SetThreshold(double threshold)
{
	this->theta = threshold;
}

/// <summary>
/// ������ֹ��������
/// </summary>
/// <param name="iterations">��������</param>
void ANN::SetTermIterations(int iterations)
{
	this->max_iter = iterations;
}

/// <summary>
/// ������ֹ������
/// </summary>
/// <param name="error">������</param>
void ANN::SetTermErrorRate(double error)
{
	this->max_error = error;
}

/// <summary>
/// ѵ��������
/// </summary>
void ANN::Train()
{
	cout << "ѵ����:";
	boost::progress_timer pt;
	boost::progress_display pd(max_iter);
	for (int t = 0; t < max_iter; t++, ++pd)
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
			cout << "��������:" << t << endl;
			cout << "��ʱ:";
			trained = true;
			return;
		}
	}
	cout << "��������:" << max_iter << endl;
	cout << "��ʱ:";
	trained = true;
}

/// <summary>
/// Ԥ��
/// </summary>
/// <param name="sample">����������ӦΪά��������㵥Ԫ����ȵ�����</param>
/// <param name="response">��Ӧֵ������ά��������㵥Ԫ����ȵ�����</param>
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

/// <summary>
/// ����������
/// </summary>
/// <param name="path">�ļ�·��</param>
/// <seealso cref="ANN:Load"/>
void ANN::Save(string path)
{
	// ʹ�����ȼ�������XML�ĵ��ṹ
	ptree net, config, layer, weights_value, min_max_value;
	config.add("<xmlcomment>", "Network Configuration");
	config.add("theta", theta);
	config.add("eta", eta);
	config.add("max_iter", max_iter);
	config.add("max_error", max_error);
	layer.add("layer_num", weights.size());
	for (int l = 0; l < weights.size(); l++)
	{
		layer.add("unit_num.value", weights[l].rows);
		ptree mat_value;
		
		for (int i = 0; i < weights[l].rows; i++)
		{
			ptree row;
			for (int j = 0; j < weights[l].cols; j++)
			{
				row.add("weight", weights[l].at(j, i));
			}
			mat_value.add_child("unit", row);
		}
		mat_value.add("<xmlcomment>", "This is No." + to_string(l+1) + " Layer of Network.");
		weights_value.add_child("layer", mat_value);
	}
	for (int i = 0; i < min_and_max.cols; i++)
	{
		min_max_value.add("min.value", min_and_max.at(i, 0));
		min_max_value.add("max.value", min_and_max.at(i, 1));
	}
	net.add_child("network.config", config);
	net.add_child("network.layer_info", layer);
	net.add_child("network.weights", weights_value);
	net.add_child("network.min_max", min_max_value);
	write_xml(path, net);
}

/// <summary>
/// ����������
/// </summary>
/// <param name="path">�ļ�·��</param>
void ANN::Load(string path)
{
	// ��ȡXML�ĵ�
	ptree net, config, layer, weights_value, min_max_value;
	if (_access(path.c_str(), 0) == -1)
	{
		cerr << "�����ļ�������" << endl;
		return;
	}
	read_xml(path, net);
	config = net.get_child("network.config");
	layer = net.get_child("network.layer_info");
	weights_value = net.get_child("network.weights");
	min_max_value = net.get_child("network.min_max");

	// ��ȡ����
	this->theta = config.get_child("theta").get_value<double>();
	this->eta = config.get_child("eta").get_value<double>();
	this->max_iter = config.get_child("max_iter").get_value<double>();
	this->max_error = config.get_child("max_error").get_value<double>();
	// ��ȡ����
	int lyr_num = layer.get_child("layer_num").get_value<int>();
	// ��ȡ����Ϣ
	Mat<int> m_lyr(1, lyr_num);
	auto lyr_info = layer.get_child("unit_num");
	int i = 0;
	for (auto iter = lyr_info.begin();iter != lyr_info.end();iter++)
	{
		m_lyr[i++] = iter->second.get_value<int>();
	}
	// ���������
	this->SetLayers(m_lyr, false);
	// ����Ȩֵ
	int l = 0, j = 0;

	for (auto lyr = weights_value.begin(); lyr != weights_value.end(); lyr++)
	{
		if (lyr->first != "layer")
			continue;
		i = 0;
		for (auto unit = lyr->second.begin(); unit != lyr->second.end(); unit++, i++)
		{
			j = 0;
			for (auto wt = unit->second.begin(); wt != unit->second.end(); wt++,j++)
			{
				weights[l].at(j,i) = wt->second.get_value<double>();
			}
		}
		l++;
	}
	// ��ȡ�����Сֵ
	min_and_max = Mat<double>(2, weights[0].rows);
	auto min_val = min_max_value.get_child("min");
	auto max_val = min_max_value.get_child("max");
	auto min_iter = min_val.begin();
	auto max_iter = max_val.begin();
	for (i = 0; i < min_and_max.cols; i++)
	{
		min_and_max.at(i, 0) = min_iter++->second.get_value<double>();
		min_and_max.at(i, 1) = max_iter++->second.get_value<double>();
	}
}

/// <summary>
/// ���ذ���������������ַ���
/// </summary>
/// <returns>ѧϰ���ʡ���ֵ�����������������������Լ�����ṹ��Ϣ</returns>
string ANN::ToString()
{
	return
		"Artificial Neural Network Configuration:\n"
		"\tStudy Rate:\t\t" + to_string(eta) + "\n"
		"\tThreshold:\t\t" + to_string(theta) + "\n"
		"\tMax Iteration Times:\t" + to_string(max_iter) + "\n"
		"\tMax Error Rate:\t\t" + to_string(max_error) + "\n"
		"\tTrain Method:\t\tBackPropagation Algorithm.\n"
		"\tIs Trained:\t\t" + string(trained ? "True" : "False") + "\n"
		"\nLayers Infomation:\n\tLayers Number:\t" + to_string(weights.size()) + "\n"
		"\tUnits Count:\t" + to_string(accumulate(weights.begin(), weights.end(), 0, [](int& i, const Mat<double>& mat) ->int { return i + mat.rows; })) + "\n"
		"\tDetails:\n"
		"\t\tInput Layer:\t" + to_string(weights.front().rows) + "\n" +
		"\t\tHidden Layer:\t" + accumulate(weights.begin() + 1, weights.end() - 1, string(), [](string &s, const Mat<double>&mat) ->string { return s + to_string(mat.rows) + ","; }) + "\n"
		"\t\tOutput Layer:\t" + to_string(weights.back().rows) + "\n";

}

/* ��ʼ��Ȩ�� */
void ANN::init_weights()
{
	mt19937 gen;
	uniform_real_distribution<> urd(0,0.1);
	_Binder<_Unforced, uniform_real_distribution<>&, mt19937&> random = bind(urd, gen);
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

/* ǰ�򴫵� */
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

/* ���򴫲� */
void ANN::backward(int &idx)
{
	// ����㷴�򴫲�
	int output_layer = outputs.rows - 1;
	for (int j = 0; j < weights[output_layer].rows; j++)
	{
		double &result_j = outputs.at(j, output_layer);
		double delta_weight = (responses.at(j,idx) - result_j) * result_j *(1 - outputs.at(j, output_layer));
		for (int k = 0; k < weights[output_layer].cols; k++)
		{
			double d = eta * delta_weight * outputs.at(k, output_layer - 1);
			weights[output_layer].at(k, j) += d;
			delta_weights[output_layer].at(k, j) = d;
		}
	}
	// ���ز㷴�򴫲�
	for (int i = output_layer - 1; i > 0; i--)
	{
		//����
		for (int j = 0; j < weights[i].rows; j++)
		{
			double sum = .0;
			// ����
			for (int k = 0; k < weights[i + 1].rows; k++)
			{
				sum += delta_weights[i + 1].at(j, k) * weights[i + 1].at(j, k); //���������й�Ȩֵ����
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

/* ��һ��ѵ���������� */
void ANN::normalize()
{

	min_and_max = Mat<double>(2, samples.cols);
	for (int i = 0; i < samples.cols; i++)
	{
		min_and_max.at(i, 0) = FLT_MAX;
		min_and_max.at(i, 1) = FLT_MIN;
	}
	// ��ֵ
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
	// ��ѵ�����ݹ�һ������
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



