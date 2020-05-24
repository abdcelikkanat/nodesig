#include <iostream>
#include <string>
#include "Graph.h"
#include "Matrix.h"
#include "src/Matrix.cpp"
#include "Model.h"
#include "src/Model.cpp"

using namespace std;



int main() {
    std::cout << "Hello, World!" << std::endl;

    //string dataset_path = "/Users/abdulkadir/workspace/nodesig/cplusplus/tests/karate.edgelist";
    string dataset_path = "/Users/abdulkadir/workspace/datasets/Homo_sapiens_undirected.edgelist";
    string dataset_path2 = "/Users/abdulkadir/workspace/nodesig/cplusplus/tests/output.edgelist";

    bool verbose = true;
    bool directed = false;
    unsigned int dim = 128;
    unsigned int walkLen = 5;
    string embFilePath = "/Users/abdulkadir/workspace/nodesig/cplusplus/deneme.embedding";

    Graph g = Graph(directed);
    g.readEdgeList(dataset_path, verbose);
    //g.writeEdgeList(dataset_path2, true);
    int unsigned numOfNodes = g.getNumOfNodes();
    int unsigned numOfEdges = g.getNumOfEdges();

    /*
    sparseMatrix<float> temp(4,4,1);
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            temp.insert(i, j,i*4+j+1);
    temp.print(1);
    auto temp2 = temp.transpose();
    temp2.print(1);
    */

    cout << numOfNodes << " " << numOfEdges << endl;
    // Get getAdjacencyMatrix
    sparseMatrix <float> adjMat = g.getAdjacencyMatrix<float>(verbose);
    // Get the row sums
    vector<Triplet<float>> rowSums = adjMat.getRowSums();
    // Normalize the rows
    cout << "Row sums ok!" << endl;
    adjMat.scaleRows(rowSums);
    // Get an identity matrix
    cout << "Scaling ok!" << endl;
    sparseEye<float> P(numOfNodes, numOfNodes);
    // Zero matrix
    sparseZero<float> sumMat(g.getNumOfNodes());
    // Get the transposed transition matrix
    sparseMatrix <float> transposedTransMat = adjMat.transpose();
    for(unsigned int l=0; l<walkLen; l++) {
        cout << "Iter: " << l << endl;
        P.rowWiseMultiply(transposedTransMat);
        sumMat.add(P);
    }


    // Define the model
    Model<float> m(numOfNodes, dim);
    // Get the data matrix elements
    //vector<Triplet<float>> &x = adjMat.getElements();
    vector<Triplet<float>> x = sumMat.getElements();
    // Encode all of them and write the embeddings into a file.
    m.encodeAll(x, embFilePath);

    return 0;
}


/*
// Driver Code
int main()
{

    // create two sparse matrices and insert values
    sparse_matrix a(4, 4);
    sparse_matrix b(4, 4);

    a.insert(1, 2, 10);
    a.insert(1, 4, 12);
    a.insert(3, 3, 5);
    a.insert(4, 1, 15);
    a.insert(4, 2, 12);
    b.insert(1, 3, 8);
    b.insert(2, 4, 23);
    b.insert(3, 3, 9);
    b.insert(4, 1, 20);
    b.insert(4, 2, 25);

    // Output result
    cout << "Addition: ";
    a.add(b);
    cout << "\nMultiplication: ";
    a.multiply(b);
    cout << "\nTranspose: ";
    sparse_matrix atranspose = a.transpose();
    atranspose.print();
}

// This code is contributed
// by Bharath Vignesh J K
*/