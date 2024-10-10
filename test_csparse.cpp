/*==============================================================================
 *     File: test_csparse.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Basic test of my CSparse implementation.
 *
 *============================================================================*/

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>

#include "csparse.h"

using namespace std;


int main(void)
{
    COOMatrix A;
    cout << "A has " << A.nnz() << " entries." << endl;
    std::array<csint, 2> A_shape = A.shape();
    cout << "A has shape (" << A_shape[0] << ", " << A_shape[1] << ")" << endl;

    // See Davis pp 7-8, Eqn (2.1)
    std::vector<csint>  i = {2,    1,    3,    0,    1,    3,    3,    1,    0,    2};
    std::vector<csint>  j = {2,    0,    3,    2,    1,    0,    1,    3,    0,    1};
    std::vector<double> v = {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7};
    COOMatrix B(v, i, j);
    cout << "B has " << B.nnz() << " entries." << endl;
    cout << "B can hold " << B.nzmax() << " entries." << endl;
    cout << "B has shape (" << B.shape()[0] << ", " << B.shape()[1] << ")" << endl;

    // Print the internals via getters
    cout << "Printing B arrays..." << endl;
    std::vector<csint> Brow = B.row();
    std::vector<csint> Bcol = B.column();
    std::vector<double> Bdata = B.data();
    for (int k = 0; k < B.nnz(); k++) {
        cout << "(" << Brow[k] << ", " << Bcol[k] << "): " << Bdata[k] << endl;
    }

    cout << "done." << endl;
    // Brow[0] = 99;  // allowed, but no effect on B since getters return a copy

    // Test printing
    B.print();
    B.print(cout, true);
    cout << B;

    // Assign an existing element
    B.assign(3, 3, 56.0);
    cout << B;
    cout << "B has " << B.nnz() << " entries." << endl;
    cout << "B can hold " << B.nzmax() << " entries." << endl;
    cout << "B has shape (" << B.shape()[0] << ", " << B.shape()[1] << ")" << endl;

    // Assign a new element that changes the dimensions
    B.assign(4, 3, 69.0);
    cout << B;
    cout << "B has " << B.nnz() << " entries." << endl;
    cout << "B can hold " << B.nzmax() << " entries." << endl;
    cout << "B has shape (" << B.shape()[0] << ", " << B.shape()[1] << ")" << endl;

    // Tranpose
    COOMatrix B_transpose = B.T();  // copy
    cout << "B.T() by copy = " << endl << B_transpose;
    assert (&B != &B_transpose);

    // Make new for given shape and nzmax
    COOMatrix D(56, 37);
    cout << "D = \n" << D;
    cout << "D can hold " << D.nzmax() << " entries." << endl;

    COOMatrix E(56, 37, (int) 1e4);
    cout << "E = \n" << E;
    cout << "E can hold " << E.nzmax() << " entries." << endl;

    // Read from a file
    std::ifstream fp("./data/t1");
    COOMatrix F(fp);
    cout << "F = \n" << F;

    // Test conversion
    B = COOMatrix(v, i, j);
    cout << "B = \n" << B;

    CSCMatrix C = B.tocsc();
    // cout << "C = \n" << C;

    return 0;
}

/*==============================================================================
 *============================================================================*/
