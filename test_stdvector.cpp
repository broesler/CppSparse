/*==============================================================================
 *     File: test_stdvector.cpp
 *  Created: 2024-10-03 12:16
 *   Author: Bernie Roesler
 *
 *  Description: 
 *
 *============================================================================*/

#include <iostream>
#include <vector>

using namespace std;

int main(void)
{
    vector<int> v = {1, 2, 3};
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;
    v.push_back(4);
    cout << "v.size() = " << v.size() << endl;  // == 4
    cout << "v.capacity() = " << v.capacity() << endl;  // == 6 auto doubles!!
    v.reserve(10);
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;
    v.push_back(5);
    cout << "v = [";
    for (auto& x : v)
        cout << x << ", ";
    cout << "]" << endl;
    return 0;
}

/*==============================================================================
 *============================================================================*/
