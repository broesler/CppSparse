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


template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> v)
{
    os << "v = [";
    for (int i = 0; i < v.size(); i++) {
        os << v[i];
        if (i < v.size() - 1) 
            os << ", ";
    }
    os << "]";

    return os;
}


int main(void)
{
    vector<int> v = {1, 2, 3};
    cout << v << endl;
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "\nv.push_back(4)" << endl;
    v.push_back(4);
    cout << v << endl;
    cout << "v.size() = " << v.size() << endl;  // == 4
    cout << "v.capacity() = " << v.capacity() << endl;  // == 6 auto doubles!!

    cout << "\nv.reserve(10)" << endl;
    v.reserve(10);
    cout << v << endl;
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "\nv.push_back(5)" << endl;
    v.push_back(5);
    cout << v << endl;
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "\nv.resize(3)" << endl;
    v.resize(3);
    cout << v << endl;
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "\nv.shrink_to_fit()" << endl;
    v.shrink_to_fit();
    cout << v << endl;
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "\nv.resize(7)" << endl;
    v.resize(7);
    cout << v << endl;
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "\nv.shrink_to_fit()" << endl;
    v.shrink_to_fit();
    cout << v << endl;
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    cout << "\nv.insert(2, 99)" << endl;
    v.insert(v.begin() + 2, 99);
    cout << v << endl;
    cout << "v.size() = " << v.size() << endl;
    cout << "v.capacity() = " << v.capacity() << endl;

    return EXIT_SUCCESS;
}

/*==============================================================================
 *============================================================================*/
