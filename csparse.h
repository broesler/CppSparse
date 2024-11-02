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

typedef std::uint64_t csint;  // TODO just change to "int" for simplicity?

#include "utils.h"
#include "csc.h"  // must be first for conversion
#include "coo.h"

#endif

//==============================================================================
//==============================================================================
