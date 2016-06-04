#include "ann.h"
using namespace std;
int main()
{
	Mat<int> mat(3, 2);
	mat.fill(3);
	for (int i = 0; i < mat.row; i++)
	{
		for (int j = 0; j < mat.col; j++)
		{
			mat.at(j, i) = i + j;
			cout << mat.at(j, i) << endl;
		}
	}
	system("pause");
}

ANN::ANN()
{
}

ANN::~ANN()
{
}
