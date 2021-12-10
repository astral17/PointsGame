#pragma once

#include "field.h"

#define DLLEXPORT extern "C" __declspec(dllexport)

DLLEXPORT Field* Init(int width, int height, int seed);