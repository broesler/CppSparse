# Error in AMD
for ash219 matrix, ATA ordering,
cs_amd finds k == 1 when nel == 60, so pk2 - pk1 == 6 here.
We never get k == 1. When nel == 60, k == 80, and the arrays are
already broken.

Q: how do we find the first point when the arrays differ from cs_amd?
A: print statements?

```C++
std::cout << "nel = " << nel
    << ", k = " << k 
    << ", C.p_[k] = " << C.p_[k]
    << ", pk2 - pk1 = " << pk2 - pk1 
    << ", degree[k] = " << degree[k]
    << ", len[k] = " << len[k]
    << ", elenk = " << elenk
    << std::endl;
```

# Debugging
First line they differ:
cs_amd.out correct values:
nel = 23, k = 61, C.p_[k] = 341, pk2 - pk1 = 4, degree[k] = 4, len[k] = 3, elenk = 1

* dbstop at start of whlie loop, when nel == 22.
* Only 1 value in `head` differs:
    cs_amd: head[4] = 61 vs.
       amd: head[4] = 80
* last head assignment?

* *many* values in `next` are incorrect, particularly, in amd:
    next[80] = 80, vs. in cs_amd, next[80] = 61.
* *many* values in `last` arrays differ, in the same way that `next` does.

* degree arrays are the same.
* `hhead` are the same (all -1)
* pk1 and pk2 are the same.
* C.p_ arrays are the same
* C.i_ arrays are the same up to element 478. C is malloc'd up to t = 695
  elements though, so those extra slots are just uninitialized memory in cs_amd.


## head and next arrays
Idea: print the `head` and `next` arrays at the start of the while loop, and
compare them.

First difference:
                                                                                                                                                   vv
   idx:                [ 0,  1,  2, 3, 4, 5,  6, 7,  8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]
cs_amd: nel = 5, next: [-1, -1, -1, 1, 2, 3, -1, 6, -1, 5,  8,  9,  4,  7, 10, 11, 13, 14, 15, 18, 17, 19, 21, 20,  0, 16, 23, 22, 25, 28, 29, 24, 75, -1, 27, 34, 35, 80, -1, 30, 31, 39, 41, 36, 26, 43, 45, 42, 47, 46, 48, 49, 40, 50, 53, 51, 54, 56, 44, 52, 33, 55, -1, 32, 59, 68, 73, 62, 63, 64, 61, 69, 57, 84, 67, 77, 74, 82, 71, 70, 72, 12, 78, 81, 83, -1]
   amd: nel = 5, next: [-1, -1, -1, 1, 2, 3, -1, 6, -1, 5,  8,  9,  4,  7, 10, 11, 13, 14, 15, 18, 17, 19, 21, 20,  0, 16, 23, 22, 25, 28, 29, 24, 68, -1, 27, 34, 35, 80, -1, 30, 31, 39, 41, 36, 26, 43, 45, 42, 47, 46, 48, 49, 40, 50, 53, 51, 54, 56, 44, 52, 33, 55, -1, 32, 59, 68, 73, 62, 75, 64, 61, 69, 57, 84, 67, 77, 74, 82, 71, 70, 72, 12, 78, 81, 83, -1]
                                                                                                                                                   ^^

idx = 32 at only difference
* find lines where next is updated, and see which ones is at index 32
    * FOUR lines in loop where next[i] is set.
    * last is often set with last[next[i]], which makes sense why the errors are
      in a similar pattern as next.
* dbstop at top of loop when nel == 4 (last known agreement of head and next)

* amd:481 when nel == 4 matches cs_amd with next[32] = 68

* cs_amd reaches this line when nel == 5:
next[last[i]] = next[i];  // cs_amd.c:198

--- cs_amd:
nel = 5
k = 66
next[k] = 73

k1 = 1
k2 = 2
ln = 2

pj = 366
i = C.i_[pj++] = 68
pj = 367
last[i] = 32
next[i] = 75

--- amd:
nel = 5
k = 66
next[k] = 73

k1 = 1
k2 = 2
ln = 2

pj = 366
i = C.i_[pj++] = 68
pj = 367
last[i] = -1   XXX
next[i] = 75

 after nel == 4 in fillreducing.cpp:481
before nel == 5 in fillreducing.cpp:292,
the last[68] is set to -1 instead of 32.

## last array
`last` is actually wrong in SIX locations (see amd_last.out)
-> print `last` arrays and see where they differ.

                                                                                                                                                                                                                                                                                                                                               vv          vv
   idx:                [ 0, 1, 2, 3,  4, 5, 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]
cs_amd; nel = 0, last: [24, 3, 4, 5, 12, 9, 7, 13, 10, 11, 14, 15, 81, 16, 17, 18, 25, 20, 19, 21, 23, 22, 27, 26, 31, 28, 37, 32, 29, 30, 39, 40, 34, 60, 35, 36, 43, 44, -1, 41, 52, 42, 47, 45, 58, 46, 49, 48, 50, 51, 53, 55, 59, 54, 56, 61, 57, 72, -1, 64, -1, 63, 67, 68, 65, 66, 69, 74, 70, 71, 77, 73, 75, 78, 76, 80, -1, 79, 82, -1, -1, 83, -1, 84, -1, -1]
cs_amd: nel = 1, last: [24, 3, 4, 5, 12, 9, 7, 13, 10, 11, 14, 15, 81, 16, 17, 18, 25, 20, 19, 21, 23, 22, 27, 26, 31, 28, 37, 32, 29, 30, 39, 40, 34, 60, 35, 36, 43, 44, -1, 41, 52, 42, 47, 45, 58, 46, 49, 48, 50, 51, 53, 55, 59, 54, 56, 61, 57, 72, -1, 64, -1, 63, 67, 68, 65, 66, 69, 74, 70, 71, 79, 73, 80, 78, -1, -1, -1, -1, 82, 75, -1, 83, 77, 84, -1, -1]

   amd: nel = 0, last: [24, 3, 4, 5, 12, 9, 7, 13, 10, 11, 14, 15, 81, 16, 17, 18, 25, 20, 19, 21, 23, 22, 27, 26, 31, 28, 37, 32, 29, 30, 39, 40, 34, 60, 35, 36, 43, 44, -1, 41, 52, 42, 47, 45, 58, 46, 49, 48, 50, 51, 53, 55, 59, 54, 56, 61, 57, 72, -1, 64, -1, 63, 67, 68, 65, 66, 69, 74, 70, 71, 77, 73, 75, 78, 76, 80, -1, 79, 82, -1, -1, 83, -1, 84, -1, -1]
   amd: nel = 1, last: [24, 3, 4, 5, 12, 9, 7, 13, 10, 11, 14, 15, 81, 16, 17, 18, 25, 20, 19, 21, 23, 22, 27, 26, 31, 28, 37, 32, 29, 30, 39, 40, 34, 60, 35, 36, 43, 44, -1, 41, 52, 42, 47, 45, 58, 46, 49, 48, 50, 51, 53, 55, 59, 54, 56, 61, 57, 72, -1, 64, -1, 63, 67, 68, 65, 66, 69, 74, 70, 71, 79, 73, 80, 78, -1, -1, -1, -1, 82, -1, -1, 83, -1, 84, -1, -1]
                                                                                                                                                                                                                                                                                                                                               ^^          ^^

idx = [79, 82]
both places where cs_amd assigned a value, but
amd left them as -1 from initialization.

last[79] should be 75.
last[82] should be 77.

Put debugging stops at each line where `last` is set. Start at nel == 0.
cs_amd will set last[79] = 75, and last[82] = 77, but amd will leave them
as -1.


