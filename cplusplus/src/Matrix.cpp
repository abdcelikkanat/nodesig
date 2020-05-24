// C++ code to perform add, multiply
// and transpose on sparse matrices
#include "Matrix.h"


template <typename T>
sparseMatrix<T>::sparseMatrix(unsigned int rowSize, unsigned int columnSize, unsigned int maxNNZ)
{
    // Initialize row size
    _rowSize = rowSize;
    // Initialize column size
    _columnSize = columnSize;
    // Resize the container
    _mat.resize(sizeof(Triplet<T>)*maxNNZ);
    // The number of non-zero elements
    _nnz = 0;

}


template <typename T>
sparseMatrix<T>::sparseMatrix(unsigned int rowSize, unsigned int columnSize)
: sparseMatrix(rowSize, columnSize, rowSize+columnSize)
{


}


template <typename T>
sparseMatrix<T>::~sparseMatrix() {

    _mat.clear();

}


template <typename T>
void sparseMatrix<T>::assign(sparseMatrix &refMat) {

    if( _rowSize != refMat._rowSize || _columnSize != refMat._columnSize ) {

        cout << "The dimensions of matrices must match!" << endl;

    } else {

        // Set the tuples
        _mat = refMat._mat;

        // Set the nnz
        _nnz = refMat._nnz;

    }
}



// insert elements into sparse Matrix
template <typename T>
void sparseMatrix<T>::insert(unsigned int rowIndex, unsigned int columnIndex, T value) {

    if(value == 0) {

        cout << "Value cannot be equal to 0." << endl;
        throw 1;

    } else if (rowIndex >= _rowSize || columnIndex >= _columnSize) {

        cout << "The row or column index cannot exceed matrix dimensions!" << endl;
        throw 1;

    } else if ( _nnz > 0 && rowIndex == _mat[_nnz-1].row && columnIndex <= _mat[_nnz-1].column ) {

        cout << "You cannot add the same indices multiple times!" << endl;
        throw 1;

    } else if ( _nnz > 0 && (rowIndex < _mat[_nnz-1].row || (rowIndex == _mat[_nnz-1].row && columnIndex < _mat[_nnz-1].column))) {

        cout << "The row or column index cannot be smaller than indices of the latest element!" << endl;
        //cout << _mat[_nnz-1].row << " " << _mat[_nnz-1].column << " | " << rowIndex << " " << columnIndex << endl;
        throw 1;

    } else {

        //cout << _nnz << " " << _mat.size() << " " << (T) _mat.capacity() << endl;
        if(_nnz < this->getCapacity())
            _mat[_nnz] = Triplet<T> {rowIndex, columnIndex, value};
        else
            _mat.push_back(Triplet<T> {rowIndex, columnIndex, value});

        _nnz++;

    }

}


template <typename T>
void sparseMatrix<T>::add(sparseMatrix &refMat) {

    // if matrices don't have same dimensions
    if (_rowSize != refMat._rowSize || _columnSize != refMat._columnSize) {

        cout << "Matrix dimension must be equal to each other";

    } else {

        sparseMatrix <T> temp(_rowSize, _columnSize);
        int pos = 0, posRefMat = 0;

        while(pos < _nnz && posRefMat < refMat._nnz) {

            // if the row of the base is higher than the row of reference or if the column is higher if they are equal
            if( ( _mat[pos].row > refMat._mat[posRefMat].row ) ||
                ( _mat[pos].row == refMat._mat[posRefMat].row && _mat[pos].column > refMat._mat[posRefMat].column) ) {

                // Insert smaller value into result
                temp.insert(refMat._mat[posRefMat].row, refMat._mat[posRefMat].column, refMat._mat[posRefMat].value);

                posRefMat++;

            // For the reverse case
            } else if( (_mat[pos].row < refMat._mat[posRefMat].row) ||
                       (_mat[pos].row == refMat._mat[posRefMat].row && _mat[pos].column < refMat._mat[posRefMat].column) ) {

                // insert smaller value into result
                temp.insert(_mat[pos].row, _mat[pos].column, _mat[pos].value);

                pos++;

            } else {
                // Insert the values if their indices are equal
                int addedval = _mat[pos].column + refMat._mat[posRefMat].column;

                if (addedval != 0)
                    temp.insert(_mat[pos].row, _mat[pos].column, addedval);
                pos++;
                posRefMat++;
            }

        }

        // insert remaining elements
        while (pos < _nnz)
            temp.insert(_mat[pos].row, _mat[pos].column, _mat[pos++].value);

        while (posRefMat < refMat._nnz)
            temp.insert(refMat._mat[posRefMat].row, refMat._mat[posRefMat].column, refMat._mat[posRefMat++].value);

        // Set the temporary matrix
        this->assign(temp);

    }
}


template <typename T>
sparseMatrix<T> sparseMatrix<T>::rowWiseMultiply(sparseMatrix <T> ref) {

    if (_columnSize != ref._columnSize) {
        cout << "Matrix dimensions are not compatible!" << endl;
        throw;
    }

    unsigned int currentRow, refCurrentRow, tempPos, refTempPos;
    T sum;

    sparseMatrix <T> tempMatrix(_rowSize, ref._rowSize);

    for(unsigned int pos=0; pos < _nnz;) {

        // Current row
        currentRow = _mat[pos].row;

        for(unsigned int refPos=0; refPos < ref._nnz;) {

            // Current row of reference matrix
            refCurrentRow = ref._mat[refPos].row;

            tempPos = pos;
            refTempPos = refPos;
            sum = 0;
            while (tempPos < _nnz && _mat[tempPos].row == currentRow &&
                    refTempPos < ref._nnz && ref._mat[refTempPos].row == refCurrentRow) {

                if (_mat[tempPos].column < ref._mat[refTempPos].column) {
                    // If the row position of the reference matrix exceeds the base matrix
                    tempPos++;
                } else if (_mat[tempPos].column > ref._mat[refTempPos].column) {
                    // If the row position of the base matrix exceeds the reference matrix
                    refTempPos++;
                } else {
                    // If the row indices match
                    sum += _mat[tempPos].value * ref._mat[refTempPos].value;
                    tempPos++;
                    refTempPos++;
                }

            }
            if (sum != 0)
                tempMatrix.insert(currentRow, refCurrentRow, sum);

            // Increment the row index of the reference matrix
            while(refPos < ref._nnz && ref._mat[refPos].row == refCurrentRow)
                refPos++;

        }

        // jump to next row
        while (pos < _nnz && _mat[pos].row == currentRow)
            pos++;

    }

    return tempMatrix;

}

template <typename T>
vector <Triplet<T>> sparseMatrix<T>::getElements() {

    return this->_mat;

}

template <typename T>
vector <Triplet<T>> sparseMatrix<T>::getRow(unsigned int row) {

    vector <Triplet<T>> vect;

    for(unsigned int pos = 0; this->_mat[pos].row <= row; pos++) {
        if(this->_mat[pos].row == row)
            vect.push_back(Triplet<T>{this->_mat[pos].row, this->_mat[pos].column, this->_mat[pos].value});
    }

    return vect;

}


template <typename T>
sparseMatrix<T> sparseMatrix<T>::transpose() {

    // The matrix to store the transposed matrix
    sparseMatrix <T> transposedMatrix(_columnSize, _rowSize, this->getCapacity());

    // Transposed matrix has the same number of non-zero elements
    transposedMatrix._nnz = _nnz;

    // Variable to store column sizes, add one more additional column for computational necessities
    int *columnNnz = new int[_columnSize];

    // Set each column sum to 0
    for(int c = 0; c < _columnSize; c++)
        columnNnz[c] = 0;

    // Count the number of non-zero elements, skip the first column
    for (int n = 0; n < _nnz; n++)
        columnNnz[_mat[n].column+1]++;

    // Variable to store cumulative column sums up to certain indices
    int *cumColNnz = new int[_columnSize];
    cumColNnz[0] = 0;
    for(int c=1; c<_columnSize; c++)
        cumColNnz[c] = cumColNnz[c-1] + columnNnz[c];

    int newRowPos;
    for(unsigned int n = 0; n < _nnz; n++) {

        // Find the new row position
        newRowPos = cumColNnz[_mat[n].column];

        // Set the indices and values
        transposedMatrix._mat[newRowPos] = Triplet<T>{_mat[n].column, _mat[n].row, _mat[n].value};

        // Increase the cumulative sum by 1, since we added one more element to this column.
        cumColNnz[_mat[n].column]++;

    }

    delete [] cumColNnz;
    delete [] columnNnz;

    return transposedMatrix;
}

template <typename T>
T sparseMatrix<T>::getMinValue() {

    if(_nnz == 0)
        return 0;

    double minValue = numeric_limits<T>::max();
    for(int n=0; n<_nnz; n++) {
        if(minValue > _mat[n].value)
            minValue = _mat[n].value;
    }

    return minValue;
}

template <typename T>
T sparseMatrix<T>::getMaxValue() {

    if(_nnz == 0)
        return 0;

    double maxValue = numeric_limits<T>::min();
    for(int n=0; n<_nnz; n++) {
        if(maxValue < _mat[n].value)
            maxValue = _mat[n].value;
    }

    return maxValue;

}

template <typename T>
vector<Triplet<T>> sparseMatrix<T>::getRowSums() {

    vector<Triplet<T>> rowSums;

    for(unsigned int r=0; r<_rowSize; r++)
        rowSums.push_back(Triplet<T>{r, r, 0});

    for(int n=0; n<_nnz; n++)
        rowSums[_mat[n].row].value += _mat[n].value;

    cout << "Size:" << rowSums.size() << endl;

    return rowSums;

}


template <typename T>
vector<Triplet<T>> sparseMatrix<T>::getColumnSums() {

    vector<Triplet<T>> columnSums;

    for(int c=0; c<_rowSize; c++)
        columnSums.push_back(Triplet<T>{c, c, 0});

    for(int n=0; n<_nnz; n++)
        columnSums[_mat[n].column].value += _mat[n].value;

    return columnSums;

}


template <typename T>
void sparseMatrix<T>::scaleBy(T value) {

    for(int n=0; n<_nnz; n++)
        _mat[n].value = _mat[n].value * value;

}

template <typename T>
void sparseMatrix<T>::scaleRows(vector <Triplet<T>> &columnVec) {

    cout << "Size:" << columnVec.size() << endl;
    /*
    for(int n=0; n<this->_nnz; n++) {
        cout << columnVec[_mat[n].row].value << endl;
        _mat[n].value = _mat[n].value * columnVec[_mat[n].row].value;
    }
     */

}

template <typename T>
void sparseMatrix<T>::scaleColumns(vector <Triplet<T>> rowVec) {

    for(int n=0; n<_nnz; n++)
        _mat[n].value = _mat[n].value * rowVec[_mat[n].column].value;

}


template <typename T>
unsigned int sparseMatrix<T>::getCapacity() {

    return _mat.capacity();

}

template <typename T>
unsigned int sparseMatrix<T>::getNnz() {

    return this->_nnz;

}

template <typename T>
sparseMatrix<T> sparseMatrix<T>::getCopy() {

    sparseMatrix <T> copiedMat(this->_rowSize, this->_columnSize, this->getCapacity());

    for(int n=0; n<this->_nnz; n++)
        copiedMat.insert(this->_mat[n].row, this->_mat[n].column, this->_mat[n].value);

    copiedMat._nnz = this->_nnz;

    return copiedMat;
}

template <typename T>
sparseMatrix<T>sparseMatrix<T>::power(unsigned int pow) {

    if(this->_rowSize != this->_columnSize) {
        cout << "This method is only implemented for sparse square matrices!." << endl;
        throw 1;
    }

    sparseMatrix<T> originalMat = this->getCopy();
    originalMat.transpose();

    if (pow == 0) {
        sparseEye<T> eye(this->_rowSize);
        return eye;
    } else if (pow % 2 == 0) {
        sparseMatrix<T> tempMat = this->power(pow / 2);
        sparseMatrix<T> malMat = tempMat.rowWiseMultiply(tempMat.transpose());
        return malMat;
    } else {
        sparseMatrix<T> tempMat = this->power(pow / 2);
        sparseMatrix<T> malMat = tempMat.rowWiseMultiply(tempMat.transpose());
        sparseMatrix<T> result = this->rowWiseMultiply(malMat.transpose());
        return result;
    }
}

template <typename T>
void sparseMatrix<T>::print(bool matrixFormat) {

    cout << "The matrix of size " << _rowSize << "x" << _columnSize << " contains " << _nnz << " non-zero elements." << endl;

    if(!matrixFormat) {

        for(int i = 0; i < _nnz; i++)
            cout << "(" << _mat[i].row << ", " << _mat[i].column  << ") " << _mat[i].value  << endl;

    } else {

        unsigned int pos=0;
        for(int r = 0; r < _rowSize; r++) {

            for (int c = 0; c < _columnSize; c++) {

                if(_nnz>0 && r == _mat[pos].row  && c == _mat[pos].column ) {

                    cout << _mat[pos].value << " ";
                    pos++;

                } else {

                    cout << "0 ";
                }
            }
            cout << endl;
        }

    }

}

template <typename T>
void sparseMatrix<T>::print() {
    print(0);
}

template <typename T>
vector <Triplet<T>>& sparseMatrix<T>::getMat() {

    return _mat;

}

