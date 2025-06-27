#ifndef PERCEPTRONVER2_HPP
#define PERCEPTRONVER2_HPP

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <sstream>
#include "d_matrix.hpp"

typedef struct optimizer{
    virtual void update(d_matrix<double>& parameter, const d_matrix<double>& grad) = 0;
    virtual ~optimizer();
}opt;

struct SGD : opt{
    double lr;
    SGD(double lr) : lr(lr) {}
    void update(d_matrix<double>& parameter, const d_matrix<double>& grad) override;
};

#endif PERCEPTRONVER2_HPP