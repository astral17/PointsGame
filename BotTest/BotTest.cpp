// BotTest.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#ifdef _DEBUG
#define _GLIBCXX_DEBUG
#endif
#define NOMINMAX
#include <Windows.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>
#include "astar.cpp"
#include "field.cpp"
#include "uct.cpp"
#include "alpha_beta.cpp"
#include "bns.cpp"
#include "layers.cpp"
#include "train.cpp"
#include "mcts.h"
#include "heuristics.h"
#include "heuristics.cpp"

using namespace std;

short SetField[] =
{
//0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
//0, 0, 0, 3, 3, 0, 3, 0, 3, 0,
//0, 0, 0, 2, 2, 2, 2, 2, 0, 0,
//0, 0, 0, 2, 3, 0, 3, 0, 0, 0,
//0, 0, 0, 2, 3, 0, 0, 0, 0, 0,
//0, 3, 0, 0, 0, 3, 3, 2, 0, 0,
//0, 0, 0, 2, 3, 0, 3, 2, 0, 0,
//0, 0, 3, 2, 2, 0, 0, 2, 0, 0,
//0, 0, 0, 3, 3, 2, 2, 0, 0, 0,
//0, 0, 0, 0, 0, 0, 0, 0, 0, 0,


//0, 0, 0, 0, 0,
//0, 0, 0, 0, 0,
//0, 2, 3, 2, 0,
//0, 0, 2, 0, 0,
//0, 0, 0, 0, 0,

0, 0, 0, 0, 0,
0, 0, 2, 0, 0,
0, 3, 0, 3, 0,
0, 0, 2, 0, 0,
0, 0, 0, 0, 0,

 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 2, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

 //0, 0, 3, 0, 3, 3, 2, 2, 0, 3, 3, 0, 0, 3, 2, 2, 2, 3, 0, 3,
 //0, 3, 0, 3, 0, 0, 2, 3, 0, 2, 0, 0, 2, 3, 2, 3, 3, 9, 3, 3,
 //0, 2, 0, 3, 3, 3, 2, 0, 3, 3, 3, 2, 6, 2, 2, 3, 9, 9, 3, 0,
 //2, 0, 3, 0, 0, 2, 3, 0, 0, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 0,
 //3, 0, 0, 0, 3, 3, 3, 3, 3, 9, 3, 2, 2, 6, 2, 3, 3, 3, 3, 0,
 //3, 2, 2, 0, 0, 2, 3, 7, 3, 3, 2, 2, 6, 2, 3, 0, 3, 3, 2, 2,
 //0, 3, 0, 3, 0, 3, 3, 3, 3, 2, 2, 6, 4, 4, 2, 2, 0, 0, 3, 0,
 //3, 0, 2, 3, 3, 9, 3, 2, 3, 2, 6, 6, 6, 4, 2, 3, 3, 3, 0, 0,
 //2, 2, 0, 3, 9, 3, 3, 2, 2, 6, 6, 2, 2, 2, 2, 3, 3, 0, 3, 0,
 //0, 3, 3, 7, 3, 2, 2, 6, 6, 2, 2, 0, 3, 3, 3, 9, 3, 2, 3, 0,
 //2, 2, 2, 3, 3, 3, 3, 2, 4, 2, 3, 3, 3, 3, 9, 3, 0, 2, 2, 0,
 //3, 2, 6, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 0, 3, 3, 3,
 //3, 3, 2, 6, 2, 2, 3, 2, 3, 2, 3, 3, 3, 3, 7, 3, 2, 3, 2, 2,
 //3, 2, 6, 6, 6, 2, 3, 3, 3, 3, 5, 3, 3, 0, 3, 3, 3, 2, 2, 3,
 //2, 2, 2, 2, 2, 6, 2, 2, 3, 7, 5, 3, 3, 2, 3, 2, 2, 6, 6, 2,
 //2, 2, 2, 2, 2, 6, 6, 2, 3, 7, 3, 2, 2, 2, 2, 2, 2, 4, 2, 3,
 //2, 6, 6, 2, 2, 2, 2, 2, 3, 3, 2, 6, 2, 2, 6, 2, 2, 2, 6, 2,
 //2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2,
 //3, 2, 2, 2, 2, 2, 6, 6, 2, 2, 2, 2, 2, 2, 2, 6, 2, 3, 2, 2,
 //3, 2, 2, 2, 3, 3, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2,

 //0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 3, 9, 3, 2, 2, 0, 3, 9, 3, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 3, 9, 3, 0, 2, 8, 2, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0,
 //0, 0, 3, 9, 3, 2, 2, 8, 2, 2, 3, 0, 0, 2, 8, 2, 0, 0, 0, 0,
 //0, 0, 0, 3, 3, 3, 0, 2, 2, 3, 3, 2, 2, 8, 2, 8, 2, 0, 0, 0,
 //0, 0, 0, 0, 2, 3, 2, 2, 3, 3, 3, 3, 2, 2, 8, 2, 0, 0, 0, 0,
 //0, 2, 2, 0, 2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 0, 0, 0, 0, 3,
 //3, 3, 0, 2, 4, 2, 2, 3, 2, 6, 2, 3, 0, 0, 2, 0, 0, 0, 0, 0,
 //0, 2, 2, 4, 6, 2, 6, 2, 2, 2, 3, 3, 0, 0, 2, 0, 0, 0, 0, 0,
 //2, 2, 0, 2, 6, 6, 6, 2, 2, 3, 7, 5, 3, 3, 2, 2, 0, 0, 0, 0,
 //0, 3, 0, 3, 2, 2, 2, 2, 3, 3, 5, 3, 9, 3, 3, 2, 0, 0, 0, 0,
 //3, 0, 3, 3, 3, 3, 2, 3, 5, 5, 3, 9, 3, 5, 5, 3, 0, 0, 0, 0,
 //2, 2, 0, 3, 2, 2, 2, 3, 3, 7, 3, 3, 2, 3, 7, 3, 0, 0, 0, 0,
 //0, 3, 0, 3, 2, 0, 2, 0, 0, 3, 0, 2, 2, 0, 3, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 8, 8, 2, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 8, 8, 2, 3, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 2, 3, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 2, 3, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0,

 //0, 0, 0, 0, 0, 0, 3, 2, 2, 0, 3, 2, 2, 3, 0, 3, 2, 2, 2, 0,
 //0, 0, 3, 0, 0, 3, 3, 3, 2, 2, 0, 3, 0, 2, 2, 2, 3, 3, 3, 2,
 //3, 3, 7, 3, 3, 9, 3, 2, 2, 0, 3, 9, 3, 3, 3, 2, 2, 2, 3, 0,
 //0, 0, 3, 3, 9, 3, 0, 2, 8, 2, 3, 3, 3, 0, 2, 8, 2, 3, 3, 3,
 //0, 0, 3, 9, 3, 2, 2, 8, 2, 2, 3, 2, 0, 2, 8, 2, 8, 2, 3, 0,
 //0, 0, 0, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2, 8, 2, 8, 2, 2, 0, 0,
 //3, 2, 0, 2, 2, 3, 2, 2, 3, 3, 3, 3, 2, 2, 8, 2, 0, 3, 3, 0,
 //0, 2, 2, 8, 2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 0, 3, 0, 2, 3,
 //3, 3, 0, 2, 4, 2, 2, 3, 2, 6, 2, 3, 0, 2, 2, 0, 0, 2, 8, 2,
 //0, 2, 2, 4, 6, 2, 6, 2, 2, 2, 3, 3, 0, 0, 2, 0, 2, 8, 2, 2,
 //2, 2, 0, 2, 6, 6, 6, 2, 2, 3, 7, 5, 3, 3, 2, 2, 8, 2, 3, 2,
 //2, 3, 3, 3, 2, 2, 2, 2, 3, 3, 5, 3, 9, 3, 3, 2, 2, 3, 3, 3,
 //3, 2, 3, 3, 3, 3, 2, 3, 5, 5, 3, 9, 3, 5, 5, 3, 2, 3, 9, 3,
 //2, 2, 0, 3, 2, 2, 2, 3, 3, 7, 3, 3, 2, 3, 7, 3, 2, 2, 3, 0,
 //2, 3, 3, 3, 2, 8, 2, 2, 0, 3, 3, 2, 2, 0, 3, 3, 2, 3, 5, 3,
 //3, 3, 3, 2, 6, 2, 6, 2, 2, 2, 2, 6, 4, 2, 2, 2, 3, 3, 7, 3,
 //0, 3, 2, 2, 2, 8, 2, 8, 8, 2, 3, 5, 3, 2, 3, 3, 9, 9, 3, 0,
 //2, 3, 0, 0, 3, 2, 2, 2, 2, 3, 7, 3, 7, 3, 3, 9, 3, 3, 3, 2,
 //2, 3, 3, 3, 3, 2, 0, 3, 3, 7, 3, 7, 3, 0, 2, 3, 9, 3, 0, 0,
 //2, 2, 2, 2, 2, 2, 0, 0, 0, 3, 0, 3, 0, 0, 2, 0, 3, 2, 0, 0,

 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 3, 0, 0, 2, 3, 3, 0, 0, 0, 3, 2, 3, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 2, 3, 7, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0,
 //0, 0, 0, 0, 3, 3, 2, 2, 3, 3, 0, 2, 2, 0, 0, 0, 3, 2, 0, 0,
 //0, 0, 3, 2, 2, 2, 6, 6, 2, 3, 3, 3, 3, 0, 0, 3, 0, 2, 0, 0,
 //0, 0, 0, 3, 2, 2, 4, 2, 2, 2, 3, 0, 2, 0, 3, 0, 0, 3, 2, 0,
 //0, 0, 3, 2, 6, 2, 2, 2, 3, 2, 3, 3, 2, 0, 3, 0, 0, 3, 2, 0,
 //0, 0, 3, 0, 2, 3, 3, 3, 3, 3, 7, 7, 3, 3, 3, 0, 0, 2, 6, 2,
 //0, 0, 0, 0, 2, 0, 3, 0, 0, 3, 7, 7, 7, 7, 3, 0, 0, 3, 2, 0,
 //0, 0, 0, 0, 3, 2, 3, 0, 3, 9, 3, 7, 7, 7, 7, 3, 0, 0, 2, 0,
 //0, 0, 0, 0, 0, 2, 3, 0, 0, 3, 2, 3, 7, 7, 7, 3, 0, 2, 0, 0,
 //0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 3, 7, 3, 0, 0, 2, 3, 0,
 //0, 0, 0, 0, 0, 3, 0, 2, 6, 2, 3, 3, 0, 3, 0, 0, 2, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 3, 0, 2, 3, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0,
 //0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0,

	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,2,2,3,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,2,6,2,3,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,2,3,2,2,0,0,0,2,0,0,0,0,0,0,0,
	//0,0,0,0,2,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//0,0,0,0,0,3,3,2,0,0,0,0,0,0,0,0,0,0,0,0,
//0,0,3,3,3,2,2,2,2,2,0,2,3,0,3,0,0,0,0,0,
//0,0,0,0,2,2,0,3,0,2,3,3,2,2,0,0,0,0,0,0,
//0,0,0,2,2,0,0,3,0,3,2,2,6,2,3,0,0,0,0,0,
//0,3,2,6,2,3,3,9,3,3,2,2,2,2,3,0,0,0,0,0,
//0,3,2,6,6,2,0,3,2,2,2,3,3,3,2,3,0,0,0,0,
//3,2,2,2,2,2,3,3,2,3,3,9,3,0,2,3,0,0,0,0,
//3,3,2,2,3,3,7,7,3,3,9,9,9,3,2,0,0,0,0,0,
//0,2,3,3,3,5,7,7,7,3,9,9,3,2,6,2,0,0,0,0,
//0,2,0,0,0,3,3,7,3,2,3,3,0,3,2,0,0,0,0,0,
//0,0,0,2,0,0,0,3,3,2,2,3,2,0,2,0,0,0,0,0,
//0,0,0,0,2,2,0,3,2,6,2,2,6,2,0,0,0,0,0,0,
//0,0,0,0,0,0,2,2,0,2,3,2,2,3,0,0,0,0,0,0,
//0,0,0,0,0,0,0,0,0,0,0,3,2,2,0,0,0,0,0,0,
//0,0,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,
//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

class HTimer
{
	LARGE_INTEGER frequency, start, end;
public:
	HTimer()
	{
		if (::QueryPerformanceFrequency(&frequency) == FALSE)
			throw "foo";
	}
	void Start()
	{
		::QueryPerformanceCounter(&start);

	}
	double Stop()
	{
		::QueryPerformanceCounter(&end);
		return static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	}
};

double Battle(function<Move(Field&)> bot1, function<Move(Field&)> bot2, mt19937 &gen, int height = 10, int width = 10, bool log_field = false)
{
	double time1 = 0, time2 = 0;
	HTimer timer;
	bool swapped = gen() & 1;
	if (swapped)
		std::swap(bot1, bot2);
	Field field(height, width);

	Move move;
	while (!field.GetAllMoves().empty())
	{
		timer.Start();
		move = bot1(field);
		(field.GetPlayer() == 0 ? time1 : time2) += timer.Stop();
		// TODO: Fix bug
		if (move == (Move)-1 || !field.CouldMove(move))
			throw;
		field.MakeMove(move);
		std::swap(bot1, bot2);
	}
	if (log_field)
		field.DebugPrint(cout, to_string(field.score), true, true);
	if (swapped)
		std::swap(time1, time2);
	std::cout << "time: " << time1 << " vs " << time2 << "\n";
	//return field.GetScore(swapped);
	int score = field.GetScore(swapped);
	if (score > 0)
	    return 1;
	if (score < 0)
	    return 0;
	return 0.5;
}

int main()
{
	//auto stime = clock();
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	LARGE_INTEGER stime;
	QueryPerformanceCounter(&stime);
	//NNetwork* net = (NNetwork*)malloc(sizeof(NNetwork));
	//Field field(5, 5);
	//mt19937 gen;
	//for (int i = 0; i < field.height; i++)
	//	for (int j = 0; j < field.width; j++)
	//		field.field[field.ToMove(j, field.height - 1 - i)] = SetField[i * field.width + j];
	//field.DebugPrint(cout, "Before", true, true);
	//Move m_move = Mcts(field, *net, gen, 1000);
	////Move m_move = Uct(field, gen, 100);
	//field.MakeMove(m_move);
	//field.DebugPrint(cout, "After", true, true);
	//Trainer();
	//ifstream fin("weights_last.bwf", ios::binary);
	//float x;
	//while (fin.read((char*)&x, 4))
	//{
	//	if (x < -2 || x > 2 || isnan(x))
	//		cout << x << " ";
	//}
	//return 0;
	//Field field(3, 3, 1);
	//field.MakeMove(field.ToMove(1, 1));
	//field.MakeMove(field.ToMove(1, 2));
	//field.MakeMove(field.ToMove(1, 0));
	//field.MakeMove(field.ToMove(2, 1));
	//field.MakeMove(field.ToMove(2, 2));
	//field.MakeMove(field.ToMove(0, 1));
	//Field field(20, 20);
	//Field field(20, 20);
	//Field field(11, 10, 1);
	Field field(5, 5, 1);
	//Field field(20, 20, 1);
	for (int i = 0; i < field.height; i++)
		for (int j = 0; j < field.width; j++)
			field.field[field.ToMove(j, field.height - 1 - i)] = SetField[i * field.width + j] | ((SetField[i * field.width + j] & 1) * 64);
	//field.MakeMove(field.ToMove(2, 2));
	//field.MakeMove(field.ToMove(2, 3));
	//field.MakeMove(field.ToMove(1, 3));
	//field.MakeMove(field.ToMove(1, 2));
	//field.MakeMove(field.ToMove(2, 4));
	//field.MakeMove(field.ToMove(3, 2));
	//field.MakeMove(field.ToMove(3, 3));
	//field.MakeMove(field.ToMove(2, 1));
	//field.MakeMove(field.ToMove(0, 0));
	//field.MakeMove(field.ToMove(3, 0));
	//field.MakeMove(field.ToMove(0, 1));
	//field.MakeMove(field.ToMove(4, 1));
	//field.DebugPrint(cout, "", true, true);
	//float input[6][5][5];
	//FieldToNNInput(field, input);
	//for (int i = 0; i < 6; i++)
	//{
	//	for (int j = 4; j >= 0; j--)
	//	{
	//		for (int k = 0; k < 5; k++)
	//			cout << input[i][j][k] << " ";
	//		cout << "\n";
	//	}
	//	cout << "\n";
	//}
	//return 0;
	//field.MakeMove(field.ToMove(4, 10));
	//field.MakeMove(field.ToMove(8, 8));
	//field.score = 24;
	//field.MakeMove(field.ToMove(11, 2));
	//MoveList moves = field.GetAllMoves();
	//for (Move move : moves)
	//{
	//	field.MakeMove(move);
	//	if (abs(field.score) > 5)
	//		cout << move << "\n";
	//	field.Undo();
	//}

	//int move = 114;
	//cout << field.ToMove(15, 5) << "\n";
	//cout << (int)field.field[move] << "\n";
	//cout << field.ToX(move) << " " << field.ToY(move) << "\n";
	
	//field.MakeMove(field.ToMove(10, 10));
	//field.MakeMove(field.ToMove(10, 9));
	//field.MakeMove(field.ToMove(11, 9));
	//field.MakeMove(field.ToMove(11, 18));
	//field.MakeMove(field.ToMove(12, 10));
	//field.MakeMove(field.ToMove(12, 9));
	//field.MakeMove(field.ToMove(11, 11));
	//field.MakeMove(field.ToMove(11, 10));
	mt19937 mt(time(0));
	//for (int i = 0; i < 100; i++)
	//cout << Uct(field, mt, 100000) << "\n";
	//return 0;
	//cout << Uct(field, mt, 450000) << "\n";
	//cout << AlphaBeta(field, 3);
	//cout << Uct(field, mt, 10000) << "\n";
	//Move move = AlphaBeta(field, mt, 6);
	//*
	//Move move = Uct(field, mt, 1000000); // Used time: 34.0991//*/
	/*
	field.MakeMove(field.ToMove(10, 10));
	field.MakeMove(field.ToMove(10, 9));
	field.MakeMove(field.ToMove(11, 9));
	field.MakeMove(field.ToMove(11, 18));
	field.MakeMove(field.ToMove(12, 10));
	field.MakeMove(field.ToMove(12, 9));
	field.MakeMove(field.ToMove(11, 11));
	field.MakeMove(field.ToMove(11, 10));
	Move move = Uct(field, mt, 10000000); // Used time: 42.039	 40.9766//*/
	//Move move = AlphaBeta(field, mt, 6);
	//field.MakeMove(394);
	//Move move = BestNodeSearch(field, mt, 6);
	//cout << move << "\n";
	//field.MakeMove(field.ToMove(3, 4));
	//field.MakeMove(Uct(field, mt, 10000));
	HeuristicStrategy hs;
	//Move move = hs.MakeMove(field, mt);
	//std::cout << move << " " << field.ToX(move) << " " << field.ToY(move) << "\n";
	//field.MakeMove(move);
	//field.MakeMove(hs.MakeMove(field, mt));
	//field.MakeMove(AlphaBeta(field, mt, 6));
	//field.MakeMove(BestNodeSearch(field, mt, 6));
	//field.DebugPrint(std::cout, to_string(field.score), true, true);
	//return 0;
	//for (int i = 0; i < 6; i++)
	//{
	//	Move move = AlphaBeta(field, mt, 6 - i / 2);
	//	cout << move << "\n";
	//	field.MakeMove(move);
	//	field.DebugPrint(cout, to_string(field.score), true, true);
	//}
	//cout << field.ToMove(3, 4) << "\n";
	//field.DebugPrint();
	//cout << "  : ";
	//for (int i = 0; i < field.width; i++)
	//	cout << setw(2) << i << " ";
	//cout << "\n";
	//for (int i = field.height - 1; i >= 0; i--)
	//{
	//	cout << setw(2) << i << ": ";
	//	for (int j = 0; j < field.width; j++)
	//		cout << setw(2) << (int)field.field[field.ToMove(j, i)] << " ";
	//	cout << "\n";
	//}
	//int score = 0;
	//field.DebugPrint(cout, to_string(field.score), true, true);
	double total_score = 0;
	int total_games = 1e6;
	for (int i = 0; i < total_games; i++)
	{
		double score = Battle(
			[&](Field& field)
			{
				//auto moves = field.GetAllMoves();
				//std::uniform_int_distribution<> d(0, moves.size() - 1);
				//return moves[d(mt)];
				//return hs.MakeMove(field, mt);
				//return AlphaBeta(field, mt, 5);
				//return field.GetAllMoves()[0];
				return Uct(field, mt, 1000);
			},
			[&](Field& field)
			{
				//auto moves = field.GetAllMoves();
				//std::uniform_int_distribution<> d(0, moves.size() - 1);
				//return moves[d(mt)];
				return hs.MakeMove(field, mt);

				//field.DebugPrint(std::cout, to_string(field.score), true, true);
				//int x, y;
				//while (true)
				//{
				//	std::cout << "Your move: ";
				//	cin >> x >> y;
				//	if (x != -1)
				//		break;
				//	x = y;
				//	cin >> y;
				//	std::cout << field.ToMove(x, y) << "\n";
				//}
				//return field.ToMove(x, y);

				//return Uct(field, mt, 200);
			}, mt, 10, 10
		);
		//std::cout << i << ": " << score << "\n";
		total_score += score;
		//if (i % 100 == 99)
		{
			std::cout << "after " << i + 1 << ": " << total_score / (double)(i + 1) << "\n";
		}
	}
	std::cout << "total: " << total_score / (double)total_games << "\n";
	/*
	HeuristicStrategy hs;
	while (!field.GetAllMoves().empty())
	{
		if (field.player == 0)
		{
			//field.MakeMove(Uct(field, mt, 1000000));
			//field.MakeMove(Uct(field, mt, 1));
			//field.MakeMove(AlphaBeta(field, mt, 6));
			auto moves = field.GetAllMoves();
			std::uniform_int_distribution<> d(0, moves.size() - 1);
			field.MakeMove(moves[d(mt)]);
		}
		else
		{
			field.MakeMove(hs.MakeMove(field, mt));
			//field.MakeMove(Uct(field, mt, 10000));
			//field.MakeMove(AlphaBeta(field, mt, 4));
		//}
		//Uct(&field, &mt, 2000000);
			//field.DebugPrint(std::cout, to_string(field.score), true, true);
			//int x, y;
			//while (true)
			//{
			//	std::cout << "Your move: ";
			//	cin >> x >> y;
			//	if (x != -1)
			//		break;
			//	x = y;
			//	cin >> y;
			//	std::cout << field.ToMove(x, y) << "\n";
			//}
			//field.MakeMove(field.ToMove(x, y));
		}
		//else
		//{
		//	field.MakeMove(uct(&field, &mt, 2000000));
		//}
		//if (field.score != score)
		//{
			//field.DebugPrint(cout, to_string(field.score), true, true);
		//	score = field.score;
		//}
	}//*/
	//cout << (int)field.player << " " << field.GetScore(1) << "\n";
	//field.DebugPrint(cout, to_string(field.GetScore(field.player)), false);
	//field.DebugPrint(cout, to_string(field.score), true, true);
	//cout << "Used time: " << (clock() - stime) / (double)CLOCKS_PER_SEC << "\n";
	LARGE_INTEGER end;
	QueryPerformanceCounter(&end);
	double interval = static_cast<double>(end.QuadPart - stime.QuadPart) / frequency.QuadPart;
	std::cout << "Used time: " << interval << "\n";
}
