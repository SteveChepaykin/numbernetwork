 #include "network.h"

void Network::Init(dataNetwork data)
{
    actFun.set();
    srand(time(NULL));
    sloi = data.length;
    size = new int[sloi];
    for (int i = 0; i < sloi; i++)
    {
        size[i] = data.size[i];
    }
    weights = new Matrix[sloi - 1];
    dops = new double *[sloi - 1];
    for (int i = 0; i < sloi - 1; i++)
    {
        // cout << sloi << endl;
        cout << size[(i+1)] << " " << size[i] << endl;
        weights[i].Init(size[i + 1], size[i]);
        dops[i] = new double[size[i + 1]];
        weights[i].Rand();
        for (int j = 0; j < size[i + 1]; j++)
        {
            dops[i][j] = (rand() % 50) * 0.06 / (size[i] + 15);
        }
    }
    neurons_values = new double *[sloi];
    neurons_errors = new double *[sloi];
    for (int i = 0; i < sloi; i++)
    {
        neurons_values[i] = new double[size[i]];
        neurons_errors[i] = new double[size[i]];
    }
    neurons_with_dops_values = new double[sloi - 1];
    for (int i = 0; i < sloi; i++)
    {
        neurons_with_dops_values[i] = 1;
    }
}

double Network::Forward()
{
    for (int k = 1; k < sloi; ++k)
    {
        // перемножение матрицы весов на маьрицу нейронов
        Matrix::Multi(weights[k - 1], neurons_values[k - 1], size[k - 1], neurons_values[k]);
        // прибавление матрицы дополнительных значений
        Matrix::SumVector(neurons_values[k], dops[k - 1], size[k]);
        // применение функции активации
        actFun.use(neurons_values[k], size[k]);
    }
    int pred = FindMaxINdex(neurons_values[sloi - 1]);
    return pred;
}

int Network::FindMaxINdex(double *value)
{
    double max = value[0];
    int prediction = 0;
    double tmp;
    for (int j = 1; j < size[sloi - 1]; j++)
    {
        tmp = value[j];
        if (tmp > max)
        {
            prediction = j;
            max = tmp;
        }
    }
    return prediction;
}

void Network::PrintConfiguration()
{
    cout << "Network has " << sloi << " layers\nsizes: ";
    for (int i = 0; i < sloi; i++)
    {
        cout << size[i] << " ";
    }
    cout << "\n\n";
}

void Network::PrintValues(int l)
{
    for (int j = 0; j < size[l]; j++)
    {
        cout << j << " " << neurons_values[l][j] << endl;
    }
}

void Network::SetInput(double *values)
{
    for (int i = 0; i < size[0]; i++)
    {
        neurons_values[0][i] = values[i];
    }
}

void Network::BackProp(double expect) {
    for (int i = 0; i < size[sloi - 1]; i++) {
        if(1 != int(expect)) {
            neurons_errors[sloi - 1][i] = -neurons_values[sloi - 1][i] * actFun.useDer(neurons_values[sloi - 1][i]);
        } else {
            neurons_errors[sloi - 1][i] = (1.0 - neurons_values[sloi - 1][i]) * actFun.useDer(neurons_values[sloi - 1][i]);
        }
    }
    for (int k = sloi - 2; k > 0; k--) {
        Matrix::Multi_T(weights[k], neurons_errors[k+1], size[k+1], neurons_errors[k]);
        for (int j = 0; j < size[k]; j++) {
            neurons_errors[k][j] *= actFun.useDer(neurons_values[k][j]);
        }
    }
}

void Network::UpdateWeights(double lr) {
    for (int i = 0; i < sloi - 1; ++i) {
        for (int j = 0; j < size[i + 1]; ++j) {
            for (int k = 0; k < size[i]; ++k) {
                weights[i](j, k) += neurons_values[i][k] * neurons_errors[i+1][j] * lr; 
            }
        }
    }
    for (int i = 0; i < sloi - 1; i++) {
        for (int k = 0; k < size[sloi + 1]; k++) {
            dops[i][k] += neurons_errors[i + 1][k] * lr;
        }
    }
}

void Network::SaveWeights() {
    ofstream out;
    out.open("weights.txt");
    if(!out.is_open()) {
        cout << "Error reading the weights file";
        system("pause");
    }
    for (int i = 0; i < sloi - 1; ++i) {
        out << weights[i] << " ";
    }
    for (int i = 0; i < sloi - 1; ++i) {
        for (int j = 0; j < size[i + 1]; ++j) {
            out << dops[i][j] << " ";
        }
    }
    cout << "weights saved \n";
    out.close();
}

void Network::ReadWeights() {
    ifstream in;
    in.open("weights.txt");
    if(!in.is_open()) {
        cout << "Error reading the weights file";
        system("pause");
    }
    for (int i = 0; i < sloi - 1; ++i) {
        in >> weights[i];
    }
    for (int i = 0; i < sloi - 1; ++i) {
        for (int j = 0; j < size[i + 1]; ++j) {
            in >> dops[i][j];
        }
    }
    cout << "weights read \n";
    in.close();
}