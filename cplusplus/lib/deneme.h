//
// Created by Abdulkadir Çelikkanat on 5/31/20.
//

#ifndef CPLUSPLUS_DENEME_H
#define CPLUSPLUS_DENEME_H

#include <iostream>
#include<Eigen/Sparse>

using namespace std;

class deneme {
public:
    deneme(int k);

    void carp(Eigen::SparseMatrix<float> &a, Eigen::SparseMatrix<float> &b);
};


#endif //CPLUSPLUS_DENEME_H
