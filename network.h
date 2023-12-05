#pragma once
#include "activatefunction.h"
#include "matrix.h"
#include <fstream>


using namespace std;

struct dataNetwork{
    int length;
    int* size;
};

class Network{
    int sloi;
    int* size;
    ActivationFunction actFun;
    Matrix* weights;
    double** dops;
    double** neurons_values, ** neurons_errors;
    double* neurons_with_dops_values;
public:
    void Init(dataNetwork data);
    void PrintConfiguration();
    void SetInput(double* values);
    double Forward();
    int FindMaxINdex(double* value);
    void PrintValues(int l);
    void BackProp(double e);
    void UpdateWeights(double nw);
    void SaveWeights();
    void ReadWeights();
};