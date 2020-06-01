//
// Created by Abdulkadir Çelikkanat on 5/31/20.
//

#include "deneme.h"

deneme::deneme(int k) {

    cout << "Given value: " << k << endl;
}

void deneme::carp(Eigen::SparseMatrix<float> &a, Eigen::SparseMatrix<float> &b) {

    //Eigen::SparseMatrix<float> c = a * b;
    cout <<  a * b << endl;


}