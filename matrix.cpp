#include "matrix.h"

void Matrix::Init(int row, int col) {
    //запоминаем строки и столбцы
    this->row = row; this->col = col;
    matrix = new double* [row];
    for (int i = 0; i < row; i++) {
        matrix[i] = new double[col];
    }
    //заполняем матрицу нулями
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            matrix[i][j] = 0;
        }
    }
}

void Matrix::Rand() {
    //заполняем матрицу случайными числами с случайно взятой зависимостью
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            matrix[i][j] = ((rand() % 100)) * 0.03 / (row + 35);
        }
    }
}

void Matrix::Multi_T(const Matrix& m1, const double* neuron, int n, double* c) {
    if(m1.col != n) {
        throw std::runtime_error("Error Multi \n");
    }
    //перемножение матриц - матрицы весов и матрицы нейронов
    for (int i = 0; i < m1.col; i++) {
        double tmp = 0;
        for (int j = 0; j < m1.row; j++) {
            tmp += m1.matrix[j][i] * neuron[j];
        }
        c[i] = tmp;
    }
}

void Matrix::Multi(const Matrix& m1, const double* neuron, int n, double* c) {
    if(m1.col != n) {
        throw std::runtime_error("Error Multi \n");
    }
    //перемножение матриц - матрицы весов и матрицы нейронов
    for (int i = 0; i < m1.row; i++) {
        double tmp = 0;
        for (int j = 0; j < m1.col; j++) {
            tmp += m1.matrix[i][j] * neuron[j];
        }
        c[i] = tmp;
    }
}

void Matrix::SumVector(double* a, const double* b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

double& Matrix::operator()(int i, int j) {
    return matrix[i][j];
}

std::ostream& operator << (std::ostream& os, const Matrix& m) {
    for (int i = 0; i < m.row; ++i) {
        for (int j = 0; j < m.col; j++) {
            os << m.matrix[i][j] << " ";
        }
    }
    return os;
}

std::istream& operator >> (std::istream& is, Matrix& m) {
    for (int i = 0; i < m.row; ++i) {
        for (int j = 0; j < m.col; j++) {
            is >> m.matrix[i][j];
        }
    }
    return is;
}