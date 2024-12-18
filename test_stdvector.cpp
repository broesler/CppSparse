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


void print_vec(const std::vector<int> v)
{
    cout << "v = [";
    for (auto& x : v)
        cout << x << ", ";
    cout << "]" << endl;
}


int main(void)
{
    vector<int> v = {1, 2, 3};
    print_vec(v);
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "v.push_back(4)" << endl;
    v.push_back(4);
    print_vec(v);
    cout << "v.size() = " << v.size() << endl;  // == 4
    cout << "v.capacity() = " << v.capacity() << endl;  // == 6 auto doubles!!

    cout << "v.reserve(10)" << endl;
    v.reserve(10);
    print_vec(v);
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "v.push_back(5)" << endl;
    v.push_back(5);
    print_vec(v);
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "v.resize(3)" << endl;
    v.resize(3);
    print_vec(v);
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "v.shrink_to_fit()" << endl;
    v.shrink_to_fit();
    print_vec(v);
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "v.resize(7)" << endl;
    v.resize(7);
    print_vec(v);
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "v.shrink_to_fit()" << endl;
    v.shrink_to_fit();
    print_vec(v);
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    return EXIT_SUCCESS;
}

/*==============================================================================
 *============================================================================*/
