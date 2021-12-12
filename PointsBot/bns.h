#pragma once

#include "field.h"
#include <random>

Move BestNodeSearch(Field& field, std::mt19937& gen, int depth);