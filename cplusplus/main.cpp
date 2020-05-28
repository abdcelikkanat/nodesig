#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <string>
#include "Graph.h"
#include <chrono>
#include <limits>
#include "Model.h"
using namespace std;

void print_mat(Eigen::SparseMatrix<float, Eigen::RowMajor> mat) {

    for(int i=0; i<mat.outerSize(); i++) {
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
            cout << "(" << it.row() << "," << it.col() << ") " << it.value() << endl;
        }
    }


}

void normalize(Eigen::SparseMatrix<float, Eigen::RowMajor> &mat) {

    float max_value, min_value, temp;

    max_value = 0;
    min_value = 0;

    int num_of_rows = mat.innerSize();
    int num_of_columns = mat.outerSize();

    // Find min/max
    for(int i=0; i<mat.outerSize(); i++) {
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
            temp = it.value();
            //cout << temp << endl;
            if(temp > max_value)
                max_value = temp;
            if(temp < min_value)
                min_value = temp;
        }
    }
    // App
    for(int i=0; i<mat.outerSize(); i++) {
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
            it.valueRef() = ( it.valueRef() - min_value ) / ( max_value - min_value );
        }
    }


    // Find row sums
    float *row_sums = new float[num_of_rows];

    for(int i=0; i<mat.outerSize(); i++) {
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it)
            row_sums[it.row()] += it.value();
    }
    //App
    for(int i=0; i<mat.outerSize(); i++) {
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
            if(row_sums[it.row()] != 0)
                it.valueRef() = it.valueRef() / row_sums[it.row()];
        }
    }

    delete [] row_sums;

}

void scale(Eigen::SparseMatrix<float, Eigen::RowMajor> &mat) {

    float minValue, maxValue;

    auto values = mat.coeffs();
    //for(unsigned int i=0; i<values.size(); i++)
    //    cout << values[i] << endl;
    minValue = values.minCoeff();
    maxValue = values.maxCoeff();

    // Scale values
    for(int i=0; i<mat.outerSize(); i++) {
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it)
            it.valueRef() = ( it.valueRef() - minValue ) / ( maxValue - minValue );
    }


    // Find row sums
    float rowSum;
    for(int i=0; i<mat.outerSize(); i++) {
        rowSum = 0;
        for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; it++)
            rowSum += it.value();

        if(rowSum != 0) {
            for(Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(mat, i); it; it++)
                it.valueRef() /= rowSum;
        }
    }


}

int main2() {

    int dim=4;
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;

    tripletList.push_back(T(0,2, 3));
    tripletList.push_back(T(1,1,5));
    tripletList.push_back(T(2,0, 4));
    tripletList.push_back(T(4,2, 1));

    Eigen::SparseMatrix<float> mat(5, 5);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    cout << mat << endl;

    Eigen::SparseMatrix<float> mat5 = mat;
    //mat.setIdentity();
    cout << mat << endl;


    cout << mat5 << endl;
    cout << "-----xxx-----" <<endl;

    Eigen::MatrixXf mat1(4, 3);
    mat1(0,0)=1;
    mat1(0,1)=2;
    mat1(0,2)=3;
    mat1(1,0)=1;
    mat1(1,1)=2;
    mat1(1,2)=3;
    mat1(2,0)=1;
    mat1(2,1)=2;
    mat1(2,2)=3;
    mat1(3,0)=1;
    mat1(3,1)=2;
    mat1(3,2)=3;

    /*
    Eigen::MatrixXf mat2(3, 2);
    mat2(0,0)=1;
    mat2(0,1)=4;
    mat2(1,0)=5;
    mat2(1,1)=9;
    mat2(2,0)=1;
    mat2(2,1)=4;
    */
    Eigen::SparseMatrix<float> mat2(3, 2);
    mat2.insert(0,0)=1;
    mat2.insert(1, 1)=4;
    mat2.insert(2,1)=3;


    cout <<"---------------" << endl;
    cout << mat1 << endl;
    cout << mat2.toDense() << endl;
    cout <<"---------------" << endl;

    Eigen::MatrixXf denek(4, 2);
    denek.noalias() = mat1 * mat2;
    cout << denek  << endl;


    scale(mat);

    return 0;
}


int main() {

    //string dataset_path = "/home/abdulkadir/Desktop/nodesig/cplusplus/tests/karate.edgelist";
    string dataset_path = "/Users/abdulkadir/workspace/datasets/Homo_sapiens_undirected.edgelist";
    string embFilePath = "/Users/abdulkadir/workspace/nodesig/cplusplus/deneme.embedding";

    bool verbose = true;
    bool directed = false;
    unsigned int dim = 128;
    unsigned int walkLen = 5;
    float cont_prob = 0.98;

    Graph g = Graph(directed);
    g.readEdgeList(dataset_path, verbose);
    //g.writeEdgeList(dataset_path2, true);
    int unsigned numOfNodes = g.getNumOfNodes();
    int unsigned numOfEdges = g.getNumOfEdges();

    //vector <vector <pair<unsigned int, double>>> adjList = g.getAdjList();


    typedef Eigen::Triplet<float> T;

    // Get edge triplets
    vector <Eigen::Triplet<float>> edgesTriplets = g.getEdges<float>();
    // Construct the adjacency matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> A(numOfNodes, numOfNodes);
    A.setFromTriplets(edgesTriplets.begin(), edgesTriplets.end());

    cout << "# of non-zero entries: " << A.nonZeros() << endl;

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

    auto start_time = chrono::steady_clock::now();
    for(unsigned int l=0; l<walkLen; l++) {
        cout << "Iter: " << l << endl;
        P = P * A;
        P = (cont_prob)*P + (1-cont_prob)*P0;
        X = X + P;
    }
    auto end_time = chrono::steady_clock::now();
    cout << "Matrix computation time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << endl;
    // Define the model
    Model<float> m(numOfNodes, dim);

    // Get the data matrix elements
    // -> The matrix S
    // Encode all of them and write the embeddings into a file.
    start_time = chrono::steady_clock::now();
    m.encodeAll(X, embFilePath);
    end_time = chrono::steady_clock::now();
    cout << "Encoding time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << endl;


    return 0;
}
