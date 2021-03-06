#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <string>
#include "Graph.h"
#include "mkl.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <bitset>
#include <chrono>

using namespace std;

template<typename T>
class Model {
private:
    unsigned int _dim;
    unsigned int _numOfNodes;
    Eigen::MatrixXf _weights;
public:

    Model(unsigned int numOfNodes, unsigned int dim);
    ~Model();
    //void encode(vector <Triplet <T>> x, bool *emb);
    void encodeAll(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath);
    void encodeAll2(Eigen::MatrixXf &X, string filePath);
    void encodeAll2(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath);
    random_device _rd;

};


template<typename T>
Model<T>::Model(unsigned int numOfNodes, unsigned int dim) {

    this->_dim = dim;
    this->_numOfNodes = numOfNodes;


    this->_weights.resize(this->_numOfNodes, this->_dim );
    // Sample numbers from normal distribution
    default_random_engine generator(this->_rd());
    normal_distribution<T> distribution(0.0, 1.0);
    for (int i=0; i < this->_numOfNodes; i++) {
        for (int j=0; j < this->_dim; j++) {
            this->_weights(i, j) = distribution(generator);
        }

    }

}

template<typename T>
Model<T>::~Model() {

}

/*
template<typename T>
void Model<T>::encode(vector <Triplet <T>> x, bool *emb) {

    T dimSum = 0;
    for(int d=0; d<this->_dim; d++) {
        dimSum = 0.0;
        for (int i = 0; i < x.size(); i++)
            dimSum += x[i].value * this->_weights[x[i].column][d];

        emb[d] = (dimSum > 0) ? true : false;

    }

}
 */


template<typename T>
void Model<T>::encodeAll(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath) {
    //cout << "2: " << A.row(0) << endl;
    fstream fs(filePath, fstream::out);
    if(fs.is_open()) {

        T dimSum;
        unsigned int idx = 0;
        Eigen::VectorXf nodeProd;
        bool e;
        Eigen::MatrixXf nodeProdX(_numOfNodes, _dim);
        string buffer;
        //_weights.transpose();

        fs << _numOfNodes << " " << _dim << endl;

        //cout << X.rows() << " " << X.cols() << endl;
        //cout << _weights.rows() << " " << _weights.cols() << endl;
        //auto start_time = chrono::steady_clock::now();
        //nodeProdX = X.toDense() * _weights;
        //auto end_time = chrono::steady_clock::now();
        //cout << "Matrix comp time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << endl;

        for(unsigned int node=0; node<_numOfNodes; node++) {
            Eigen::SparseVector<T, Eigen::RowMajor> nodeVect = X.row(node);
            //cout << X.row(node) << endl;
            //cout << nodeVect.rows() << " " << nodeVect.cols() << endl;
            //cout << X.rows() << " " << X.cols() << endl;
            nodeProd = nodeVect * _weights;


            //fs << node << " ";
            buffer += to_string(node) + " ";
            for(unsigned int d=0; d<_dim; d++) {
                //cout << nodeProd.coeff(d) << endl;
                if(nodeProd.coeff(d) > 0)
                    buffer += "1 "; //fs << "1 ";
                else
                    buffer += "0 "; //fs << "0 ";
            }
            buffer += "\n"; // fs << endl;

            if((node+1) % 64 == 0 || (node+1) == _numOfNodes ) {
                fs << buffer;
                buffer.clear();
            }

        }
        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}


template<typename T>
void Model<T>::encodeAll2(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath) {

    if(_dim % 8 != 0) {
        cout << "Dimension must be divisible by 8!" << endl;
        throw;
    }

    fstream fs(filePath, fstream::out | fstream::binary);
    if(fs.is_open()) {

        T dimSum;
        unsigned int idx = 0;
        Eigen::VectorXf nodeProd;
        bool e;
        Eigen::MatrixXf nodeProdX(_numOfNodes, _dim);
        string buffer;
        //_weights.transpose();

        cout << _numOfNodes << " - " << _dim << endl;
        //bitset<2*sizeof(char)> b(5);
        //unsigned int b1 = ;
        fs.write(reinterpret_cast<const char *>(&_numOfNodes), 4);
        //unsigned int b2 = 3; // number of bytes to read
        fs.write(reinterpret_cast<const char *>(&_dim), 4);





        //cout << X.rows() << " " << X.cols() << endl;
        //cout << _weights.rows() << " " << _weights.cols() << endl;


        auto start_time = chrono::steady_clock::now();
        nodeProdX = X * _weights;
        auto end_time = chrono::steady_clock::now();
        cout << "Matrix comp time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << endl;


        for(unsigned int node=0; node<_numOfNodes; node++) {
            Eigen::VectorXf nodeVect = nodeProdX.row(node);
            //cout << X.row(node) << endl;
            //cout << nodeVect.rows() << " " << nodeVect.cols() << endl;
            //cout << X.rows() << " " << X.cols() << endl;
            //nodeProd = nodeProdX.row(node); //nodeVect * _weights;
            //nodeProd = nodeVect * _weights;

            //vector<bool> bin(_dim, 0);
            vector<uint8_t> bin(_dim/8, 0);


            //fs << node << " ";
            /*
            buffer += to_string(node) + " ";
            for(unsigned int d=0; d<_dim; d++) {
                //cout << nodeProd.coeff(d) << endl;
                if(nodeVect.coeff(d) > 0)
                    buffer += "1 "; //fs << "1 ";
                else
                    buffer += "0 "; //fs << "0 ";
            }
            */
            for (unsigned int d = 0; d < _dim; d++) {
                bin[int(d/8)] <<= 1;
                if (nodeVect.coeff(d) > 0)
                    bin[int(d/8)] += 1;
            }

            copy(bin.begin(), bin.end(), std::ostreambuf_iterator<char>(fs));

        }

        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}


template<typename T>
void Model<T>::encodeAll2(Eigen::MatrixXf &X, string filePath) {

    if(_dim % 8 != 0) {
        cout << "Dimension must be divisible by 8!" << endl;
        throw;
    }

    fstream fs(filePath, fstream::out | fstream::binary);
    if(fs.is_open()) {

        T dimSum;
        unsigned int idx = 0;
        Eigen::VectorXf nodeProd;
        bool e;
        Eigen::MatrixXf nodeProdX(_numOfNodes, _dim);
        string buffer;
        //_weights.transpose();

        cout << _numOfNodes << " - " << _dim << endl;
        //bitset<2*sizeof(char)> b(5);
        //unsigned int b1 = ;
        fs.write(reinterpret_cast<const char *>(&_numOfNodes), 4);
        //unsigned int b2 = 3; // number of bytes to read
        fs.write(reinterpret_cast<const char *>(&_dim), 4);





        //cout << X.rows() << " " << X.cols() << endl;
        //cout << _weights.rows() << " " << _weights.cols() << endl;


        auto start_time = chrono::steady_clock::now();
        nodeProdX = X * _weights;
        auto end_time = chrono::steady_clock::now();
        cout << "Matrix comp time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << endl;


        for(unsigned int node=0; node<_numOfNodes; node++) {
            Eigen::VectorXf nodeVect = nodeProdX.row(node);
            //cout << X.row(node) << endl;
            //cout << nodeVect.rows() << " " << nodeVect.cols() << endl;
            //cout << X.rows() << " " << X.cols() << endl;
            //nodeProd = nodeProdX.row(node); //nodeVect * _weights;
            //nodeProd = nodeVect * _weights;

            //vector<bool> bin(_dim, 0);
            vector<uint8_t> bin(_dim/8, 0);


            //fs << node << " ";
            /*
            buffer += to_string(node) + " ";
            for(unsigned int d=0; d<_dim; d++) {
                //cout << nodeProd.coeff(d) << endl;
                if(nodeVect.coeff(d) > 0)
                    buffer += "1 "; //fs << "1 ";
                else
                    buffer += "0 "; //fs << "0 ";
            }
            */
            for (unsigned int d = 0; d < _dim; d++) {
                bin[int(d/8)] <<= 1;
                if (nodeVect.coeff(d) > 0)
                    bin[int(d/8)] += 1;
            }

            copy(bin.begin(), bin.end(), std::ostreambuf_iterator<char>(fs));

        }

        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}


/*
template<typename T>
void Model<T>::encodeAll(Eigen::SparseMatrix<T, Eigen::RowMajor> &X, string filePath) {
    //cout << "2: " << A.row(0) << endl;
    fstream fs(filePath, fstream::out);
    if(fs.is_open()) {

        T dimSum;
        unsigned int idx = 0;
        Eigen::VectorXf nodeProd;
        bool e;

        //_weights.transpose();

        fs << _numOfNodes << " " << _dim << endl;

        for(unsigned int node=0; node<_numOfNodes; node++) {
            Eigen::SparseVector<T, Eigen::RowMajor> nodeVect = X.row(node);
            //cout << X.row(node) << endl;
            //cout << nodeVect.rows() << " " << nodeVect.cols() << endl;
            //cout << X.rows() << " " << X.cols() << endl;
            nodeProd = nodeVect * _weights;

            fs << node << " ";
            for(unsigned int d=0; d<_dim; d++) {
                //cout << nodeProd.coeff(d) << endl;
                if(nodeProd.coeff(d) > 0)
                    fs << "1 ";
                else
                    fs << "0 ";
            }

            fs << endl;

        }
        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}




template<typename T>
void Model<T>::encodeAll(Eigen::Triplet<T> &x, string filePath) {

    fstream fs(filePath, fstream::out);
    if(fs.is_open()) {

        T dimSum;
        unsigned int idx = 0;
        bool e;

        fs << _numOfNodes << " " << _dim << endl;

        unsigned int nodePos;

        nodePos = 0;
        for(unsigned int node = 0; node<_numOfNodes; node++) {

            fs << node << " ";
            for (int d = 0; d < this->_dim; d++) {
                idx = nodePos;
                dimSum = 0;
                while (x[idx].row == node) {
                    dimSum += x[idx].value * this->_weights[x[idx].column][d];
                    idx++;
                }
                e = (dimSum > 0) ? 1 : 0;
                fs << e << " ";
            }
            fs << endl;
            nodePos = idx;

        }

        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}
*/

/*
template<typename T>
Model<T>::Model(unsigned int numOfNodes, unsigned int dim) {

    this->_dim = dim;
    this->_numOfNodes = numOfNodes;

    this->_weights = new T *[numOfNodes];
    for(int i=0; i < this->_numOfNodes; i++)
        this->_weights[i] = new T[this->_dim];

    // Sample numbers from normal distribution
    default_random_engine generator(this->_rd());
    normal_distribution<T> distribution(0.0, 1.0);
    for (int i=0; i < this->_numOfNodes; i++) {
        for (int j=0; j < this->_dim; j++) {
            this->_weights[i][j] = distribution(generator);
        }
    }

}

template<typename T>
Model<T>::~Model() {

    for(int i=0; i<this->_numOfNodes; i++)
        delete [] _weights[i];
    delete [] _weights;

}
 */

#endif //MODEL_H
