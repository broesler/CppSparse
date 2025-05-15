# Reachability of the Elimination Tree
## Davis, Chapter 4 *Cholesky Decomposition*

### CSparse Implementation
The CSparse function `cs_ereach` uses a stack to traverse the elimination tree. The stack is just an
array with a pointer to the top of the stack. The output stack is stored in the same array, just at
the end, giving a structure like this for `N = 11`:

```
[ stack                  | output ]
len                            top
  v                              v
[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  indices
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0]  values
```

As we traverse the stack, `len` gets incremented. Let's say we traversed nodes
1 through 4, and node 4 is a root node. The stack would look like this:

```
[ stack                  | output ]
-------> len                   top
           v                     v
[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  indices
[ 1, 2, 3, 4, 0, 0, 0, 0, 0, 0,  0]  values
```

Once a root or marked node is reached, we pop elements off of the stack and push them onto the
output stack, like this:

```
[ stack            | output       ]
len <-------         top <--------
  v                    v
[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  indices
[ 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,  4]  values
```

The same process is repeated from the next node. Let's say we traversed nodes
5 through 7. The stack would look like this:

```
[ stack                  | output ]
----> len                      top
        v                        v
[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  indices
[ 5, 6, 7, 0, 0, 0, 0, 0, 0, 0,  0]  values
```

Then the output:
```
[ stack     | output              ]
len <-------  top <-----
  v           v
[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  indices
[ 0, 0, 0, 0, 5, 6, 7, 1, 2, 3,  4]  values
```

This process means that the output stack is in reverse order of the elimination tree traversal for
each subpath, which results in a topological order of the entire output.

### C++ Implementation
In our C++ implementation `ereach`, we use a `std::vector` to store the stack and output so that we
can use more idiomatic code like `push_back` and `pop_back` to make it clear what is happening to
each stack. The downside of this design is that we end up building the output in reverse order.

The code to build the output stack is as follows:

```cpp
while (!stack.empty()) {
    output.push_back(stack.back());
    stack.pop_back();
}
```

For the first subpath traversal, we get:
```
           <---- back
                    v
stack  = [ 1, 2, 3, 4]
           ----> back
                    v
output = [ 4, 3, 2, 1]
```

with the more efficient idiomatic line:

```cpp
std::copy(stack.rbegin(), stack.rend(), std::back_inserter(output));
```

On the next subpath traversal, we get:

```
          <-- back
                 v
stack  = [ 5, 6, 7]
                    ----> back
                             v
output = [ 4, 3, 2, 1, 7, 6, 5]
```

We can see that our output is in reverse order of the array `s[top:N-1]` in the CSparse
implementation, so we return a reversed vector to get the topological order.
Reversing in-place is more efficient than copying the output to a new vector, so
we can use:

```cpp
std::reverse(output.begin(), output.end());
return output;
```

