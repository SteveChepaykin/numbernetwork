#include <cmath>
#include <iostream>
#include "activatefunction.h"

void ActivationFunction::set() {
    std::cout << "Set activation pls\n1 - sigmoid\n2 - ReLU\n3 - tg\n";
    int tmp;
    std::cin >> tmp;
    switch (tmp) {
        case sigmoid:
            actfun = sigmoid;
            break;
        case ReLU:
            actfun = ReLU;
            break;
        case thx:
            actfun = thx;
            break;
        default:
            throw std::runtime_error("Error read actFun");
            break;
    }
}

void ActivationFunction::use(double* value, int n) {
    switch (actfun)
    {
    case activateFunc::sigmoid:
        // преобразование активации сигмоида
        for(int i = 0; i < n; i++) {
            value[i] = 1 / (1  + exp(-value[i]));
        }
        break;
    case activateFunc::ReLU:
        // преобразование активации через Релу
        for(int i = 0; i < n; i++) {
            if(value[i] < 0) {
                value[i] *= 0.01;
            } else if(value[i] > 1) {
                value[i] = 1. + 0.01 * (value[i] - 1.);
            } else {
                value[i] = value[i];
            }
        }
        break;
    case activateFunc::thx:
        // Преобразование активации через тангенс
        for(int i = 0; i < n; i++) {
            if(value[i] < 0) {
                value[i] = 0.01 * (exp(value[i]) - exp(-value[i])) / (exp(value[i]) + exp(-value[i]));
            } else {
                value[i] = (exp(value[i]) - exp(-value[i])) / (exp(value[i]) + exp(-value[i]));
            }
        }
        break;
    default:
        throw std::runtime_error("Error actFun\n");
        break;
    }
}

void ActivationFunction::useDer(double* value, int n) {
    switch (actfun)
    {
    case activateFunc::sigmoid:
        // преобразование активации с производной сигмоида
        for(int i = 0; i < n; i++) {
            value[i] = value[i] * (1 - value[i]);
        }
        break;
    case activateFunc::ReLU:
        // преобразование активации с производной Релу
        for(int i = 0; i < n; i++) {
            if(value[i] < 0 || value[i] > 1) {
                value[i] = 0.01;
            } else {
                value[i] = 1;
            }
        }
        break;
    case activateFunc::thx:
        // Преобразование активации с производной тангенса
        for(int i = 0; i < n; i++) {
            if(value[i] < 0) {
                value[i] = 0.01 * (1 - value[i] * value[i]);
            } else {
                value[i] = 1 - value[i] * value[i];
            }
        }
        break;
    default:
        throw std::runtime_error("Error actFun dericative\n");
        break;
    }
}

double ActivationFunction::useDer(double value) {
    switch (actfun)
    {
    case activateFunc::sigmoid:
        value = 1 / (1 + exp(-value));
        break;
    case activateFunc::ReLU:
        if(value < 0 || value > 1) {
            value = 0.01;
        }
        break;
    case activateFunc::thx:
        if(value < 0) {
            value = 0.01 * ((exp(value) - exp(-value))/(exp(value) + exp(-value)));
        } else {
            value = (exp(value) - exp(-value))/(exp(value) + exp(-value)); 
        }
        break;
    default:
        throw std::runtime_error("Error actFun dericative\n");
        break;
    }
    return value;
}