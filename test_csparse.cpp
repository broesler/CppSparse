/*==============================================================================
 *     File: test_csparse.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Basic test of my CSparse implementation.
 *
 *============================================================================*/

#include <iostream>

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
    std::array<csint, 2> B_shape = B.shape();
    cout << "B has shape (" << B_shape[0] << ", " << B_shape[1] << ")" << endl;

    B.print();
    B.print(true);

    return 0;
}

/*==============================================================================
 *============================================================================*/
