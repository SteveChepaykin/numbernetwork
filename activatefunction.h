#pragma once
#include <iostream>

enum activateFunc { sigmoid  = 1, ReLU, thx };
class ActivationFunction {
    activateFunc actfun;
public:
    void set();
    void use(double* value, int n);
    void useDer(double* value, int n);
    //перегрузка для производной back propagation
    double useDer(double value);
};