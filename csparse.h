//==============================================================================
//    File: csparse.h
// Created: 2024-10-09 21:01
//  Author: Bernie Roesler
//
//  Description: The header file for the CSparse++ package with definitions of
//    the matrix classes and associated functions.
//
//==============================================================================

#ifndef _CSPARSE_H_
#define _CSPARSE_H_

#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using csint = std::int64_t;
using Shape = std::array<csint, 2>;

// Pre-declare classes for type conversions
class COOMatrix;
class CSCMatrix;

#include "utils.h"
#include "csc.h"
#include "coo.h"

#endif

//==============================================================================
//==============================================================================
