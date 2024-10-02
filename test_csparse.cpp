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
    return 0;
}

/*==============================================================================
 *============================================================================*/
