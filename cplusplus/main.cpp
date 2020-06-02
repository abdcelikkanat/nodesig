#define EIGEN_USE_MKL_ALL
#include "mkl.h"
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include "deneme.h"
#include "Graph.h"
#include "Model.h"

using namespace std;
using namespace Eigen;

void scale(Eigen::SparseMatrix<float, Eigen::RowMajor> &mat);

void ppmi_matrix(Eigen::MatrixXf &Mat);

int main() {

    //string dataset_path = "/Users/abdulkadir/workspace/nodesig/cplusplus/tests/karate.edgelist";
    //string dataset_path = "/Users/abdulkadir/workspace/nodesig/cplusplus/Homo_sapiens_renaissance.edgelist";
    //string embFilePath = "/Users/abdulkadir/workspace/nodesig/cplusplus/deneme.embedding";

    string dataset_path = "/home/abdulkadir/Desktop/datasets/Homo_sapiens_renaissance.edgelist";
    string embFilePath = "/home/abdulkadir/Desktop/nodesig/cplusplus/Homo_sapiens_renaissance.embedding";
    //string dataset_path = "/home/abdulkadir/Desktop/datasets/youtube_renaissance.edgelist";
    //string embFilePath = "/home/abdulkadir/Desktop/nodesig/cplusplus/youtube_renaissance.embedding";

    //string dataset_path = "/home/kadir/workspace/cplusplus/youtube_renaissance.edgelist";
    //string embFilePath = "/home/kadir/workspace/cplusplus/youtube_renaissance.embedding";


    bool verbose = true;
    bool directed = false;
    unsigned int dim = 1024*8; // 128
    unsigned int walkLen = 1;
    float cont_prob = 0.98;

    Graph g = Graph(directed);
    g.readEdgeList(dataset_path, verbose);
    //g.writeEdgeList(dataset_path2, true);
    int unsigned numOfNodes = g.getNumOfNodes();
    int unsigned numOfEdges = g.getNumOfEdges();


    typedef float T;

    // Get edge triplets
    vector <Triplet<T>> edgesTriplets = g.getEdges<T>();
    // Construct the adjacency matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> A(numOfNodes, numOfNodes);
    A.setFromTriplets(edgesTriplets.begin(), edgesTriplets.end());

    // Normalize the adjacency matrix
    cout << "Scaling started!" << endl;
    scale(A);
    cout << "Scaling completed!" << endl;

    // Construct zero matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> X(numOfNodes, numOfNodes);

    // Construct the identity matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> P(numOfNodes, numOfNodes);
    P.setIdentity();

    // Construct another identity matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> P0(numOfNodes, numOfNodes);
    P0.setIdentity();

    cout << "Compressigon started!" << endl;
    // Compress matrices
    P.makeCompressed();
    P0.makeCompressed();
    X.makeCompressed();
    cout << "Compression finished!" << endl;

    /* Random walk */
    for(unsigned int l=0; l<walkLen; l++) {
        cout << "Iter: " << l << endl;
        P = P * A;
        P = (cont_prob)*P + (1-cont_prob)*P0;
        X = X + P;
    }



    // Get the PPMI matrix
    // add a condition here to convert ppmi, burada is var
    cout << "Y is aaigned started!" << endl;
    SparseMatrix <T, RowMajor> Y(numOfNodes, numOfNodes);
    Y = X; //Y.noalias() = X.toDense();
    cout << "Y assingmened finished!" << endl;
    //ppmi_matrix(Y);

    cout << "Model is started!" << endl;
    Model<T> m(numOfNodes, dim);
    cout << "Model started!" << endl;

    //cout << mat2 * mat2 << endl;
    cout << "Encoding started!" << endl;
    m.encodeAll2(Y, embFilePath);
    cout << "Encoding finished!" << endl;

    return 0;
}


int main2() {

    //string dataset_path = "/Users/abdulkadir/workspace/nodesig/cplusplus/tests/karate.edgelist";
    //string dataset_path = "/Users/abdulkadir/workspace/nodesig/cplusplus/Homo_sapiens_renaissance.edgelist";
    //string embFilePath = "/Users/abdulkadir/workspace/nodesig/cplusplus/deneme.embedding";

    //string dataset_path = "/home/abdulkadir/Desktop/datasets/Homo_sapiens_renaissance.edgelist";
    //string embFilePath = "/home/abdulkadir/Desktop/nodesig/cplusplus/Homo_sapiens_renaissance.embedding";
    string dataset_path = "/home/abdulkadir/Desktop/datasets/youtube_renaissance.edgelist";
    string embFilePath = "/home/abdulkadir/Desktop/nodesig/cplusplus/youtube_renaissance.embedding";

    bool verbose = true;
    bool directed = false;
    unsigned int dim = 1024*8; // 128
    unsigned int walkLen = 1;
    float cont_prob = 0.98;

    Graph g = Graph(directed);
    g.readEdgeList(dataset_path, verbose);
    //g.writeEdgeList(dataset_path2, true);
    int unsigned numOfNodes = g.getNumOfNodes();
    int unsigned numOfEdges = g.getNumOfEdges();


    typedef float T;

    // Get edge triplets
    vector <Triplet<T>> edgesTriplets = g.getEdges<T>();
    // Construct the adjacency matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> A(numOfNodes, numOfNodes);
    A.setFromTriplets(edgesTriplets.begin(), edgesTriplets.end());

    // Normalize the adjacency matrix
    scale(A);

    // Construct zero matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> X(numOfNodes, numOfNodes);

    // Construct the identity matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> P(numOfNodes, numOfNodes);
    P.setIdentity();

    // Construct another identity matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> P0(numOfNodes, numOfNodes);
    P0.setIdentity();

    // Compress matrices
    P.makeCompressed();
    P0.makeCompressed();
    X.makeCompressed();

    /* Random walk */
    for(unsigned int l=0; l<walkLen; l++) {
        cout << "Iter: " << l << endl;
        P = P * A;
        P = (cont_prob)*P + (1-cont_prob)*P0;
        X = X + P;
    }



    // Get the PPMI matrix
    // add a condition here to convert ppmi, burada is var
    MatrixXf Y(numOfNodes, numOfNodes);
    Y.noalias() = X.toDense();
    ppmi_matrix(Y);

    Model<T> m(numOfNodes, dim);

    //cout << mat2 * mat2 << endl;
    m.encodeAll2(Y, embFilePath);


    return 0;
}


int main3() {

    int dim=4;
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;

    tripletList.push_back(T(0,2, 3));
    tripletList.push_back(T(1,1,5));
    tripletList.push_back(T(2,0, 4));
    tripletList.push_back(T(4,2, 1));

    Eigen::SparseMatrix<float> mat1(5, 5);
    mat1.setFromTriplets(tripletList.begin(), tripletList.end());

    Eigen::SparseMatrix<float> mat2(5, 5);
    mat2.setFromTriplets(tripletList.begin(), tripletList.end());


    cout << mat1 << endl;

    cout << "---------------" << endl;

    cout << mat2 << endl;

    cout << "---------------" << endl;

    cout << mat1 * mat2 << endl;

    cout << "---------------" << endl;

    return 0;
}

void scale(Eigen::SparseMatrix<float, Eigen::RowMajor> &mat) {

    float minValue, maxValue;

    auto values = mat.coeffs();
    //for(unsigned int i=0; i<values.size(); i++)
    //    cout << values[i] << endl;

    // Set diagonals
    auto diagonals = mat.diagonal();
    for(int d=0; d<diagonals.size(); d++) {
        diagonals.coeffRef(d, d) = 0;
        if(mat.row(d).sum() == 0)
            diagonals.coeffRef(d,d)=1;
    }

    /*
    minValue = values.minCoeff();
    maxValue = values.maxCoeff();
*/
    /*
    // Scale values
    for(int i=0; i<mat.outerSize(); i++) {
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it)
            it.valueRef() = ( it.valueRef() - minValue ) / ( maxValue - minValue );
    }
    */


    // Find row sums
    float rowSum;
    for(int i=0; i<mat.outerSize(); i++) {
        rowSum = 0;
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it)
            rowSum += it.value();

        if(rowSum != 0) {
            for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it)
                it.valueRef() = it.valueRef() / rowSum;
        }
    }

    /*
    // Find row sums
    float *col_sums = new float[mat.rows()];

    for(int i=0; i<mat.outerSize(); i++) {
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it)
            col_sums[it.col()] += it.value();
    }
    //App
    for(int i=0; i<mat.outerSize(); i++) {
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
            if(col_sums[it.col()] != 0)
                it.valueRef() = it.valueRef() / col_sums[it.col()];
        }
    }

    delete [] col_sums;
     */

}

void ppmi_matrix(Eigen::MatrixXf &Mat) {

    auto rowSum = Mat.colwise().sum();
    auto colSum = Mat.rowwise().sum();
    auto totalSum = colSum.sum();
    Eigen::MatrixXf prod(Mat.rows(), Mat.cols());
    prod = colSum * rowSum;

    for(unsigned int i=0; i<Mat.rows(); i++) {
        for(unsigned int j=0; j<Mat.cols(); j++) {

            if(prod(i,j) != 0) {
                Mat(i, j) = log(totalSum * Mat(i, j) / prod(i, j) );
                if(Mat(i, j)  < 0)
                    Mat(i, j) = 0;
            } else {
                Mat(i, j) = 0;
            }

        }
    }

}