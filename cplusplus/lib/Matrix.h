/* 0.0.1 */
#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <climits>

using namespace std;

/* */
template <class T>
struct Triplet {
    unsigned int row;
    unsigned int column;
    T value;
};


template <class T>
class sparseMatrix {
protected:
    //valType **_mat;
    vector <Triplet<T>> _mat;

    // dimensions of Matrix
    unsigned int _rowSize, _columnSize;

    // total number of elements in Matrix
    unsigned int _nnz;

public:

    sparseMatrix(unsigned int rowSize, unsigned int columnSize, unsigned int maxNNZ);
    sparseMatrix(unsigned int rowSize, unsigned int columnSize);
    ~sparseMatrix();
    void assign(sparseMatrix <T> &refMat);
    void insert(unsigned int rowIndex, unsigned int columnIndex, T value);
    void add(sparseMatrix <T>&sparseMat);
    sparseMatrix<T> transpose();
    sparseMatrix<T> rowWiseMultiply(sparseMatrix <T> ref);
    T getMaxValue();
    T getMinValue();
    void scaleBy(T value);
    void scaleRows(vector <Triplet<T>> &columnVec);
    void scaleColumns(vector <Triplet<T>> columnVec);
    vector <Triplet<T>> getRowSums();
    vector <Triplet<T>> getColumnSums();
    vector <Triplet<T>> getElements();
    vector <Triplet<T>> getRow(unsigned int row);
    unsigned int getCapacity();
    unsigned int getNnz();
    sparseMatrix <T> getCopy();
    void print(bool matrixFormat);
    void print();
    vector <Triplet<T>>& getMat();
    sparseMatrix <T> power(unsigned int pow);

};

template <class T>
class sparseEye: public sparseMatrix <T>{
public:

    sparseEye(unsigned int size, unsigned int maxNnz) : sparseMatrix <T>(size, size, maxNnz) {

        this->_nnz = size;

        cout << "Constructor " << this->_mat.size() << endl;

        for(int i=0; i<size; i++) {
            this->_mat[i].row = i;
            this->_mat[i].column = i;
            this->_mat[i].value = 1.0;
        }

    }

    sparseEye(unsigned int size) : sparseEye <T>(size, 2*size) {

    }

};

/* */
template <class T>
class sparseZero: public sparseMatrix <T>{
public:
    sparseZero(unsigned int size, unsigned int maxNnz) : sparseMatrix<T>(size, size, maxNnz) {
        this->_nnz = 0;
    }

    sparseZero(unsigned int size) : sparseMatrix <T>(size, size) {
        this->_nnz = 0;
    }

};




#endif //MATRIX_H
