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

/*
int main() {

    int dim=4;
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;

    tripletList.push_back(T(0,2, 3));
    tripletList.push_back(T(1,1,5));
    tripletList.push_back(T(2,0, 4));
    tripletList.push_back(T(1,2, 1));

    Eigen::SparseMatrix<float> mat(dim, dim);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());


    print_mat(mat);

    cout <<"---------------" << endl;
    normalize(mat);

    print_mat(mat);


    return 0;
}
*/

int main() {

    string dataset_path = "/home/abdulkadir/Desktop/nodesig/cplusplus/tests/karate.edgelist";
    //string dataset_path = "/Users/abdulkadir/workspace/datasets/Homo_sapiens_undirected.edgelist";
    string embFilePath = "/home/abdulkadir/Desktop/nodesig/cplusplus/deneme.embedding";

    bool verbose = true;
    bool directed = false;
    unsigned int dim = 128;
    unsigned int walkLen = 5;


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
    normalize(A);
    // Construct zero matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> S(numOfNodes, numOfNodes);
    // Construct the identity matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> P(numOfNodes, numOfNodes);
    vector<T> PTripletList;
    for(int i=0; i<numOfNodes; i++)
        PTripletList.push_back(T(i, i, 1));
    P.setFromTriplets(PTripletList.begin(), PTripletList.end());

    // Compress matrices
    P.makeCompressed();
    S.makeCompressed();

    auto start_time = chrono::steady_clock::now();
    for(unsigned int l=0; l<walkLen; l++) {
        cout << "Iter: " << l << endl;
        P = P * A;
        S = S + P;
    }
    auto end_time = chrono::steady_clock::now();
    cout << "Matrix computation time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << endl;
    // Define the model
    Model<float> m(numOfNodes, dim);

    // Get the data matrix elements
    // -> The matrix S
    // Encode all of them and write the embeddings into a file.
    m.encodeAll(S, embFilePath);
    /* */
    return 0;
}
