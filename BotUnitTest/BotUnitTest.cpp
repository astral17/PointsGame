#include "CppUnitTest.h"
//#include "../PointsBot/uct.h"
#include "field.cpp"
#include "uct.cpp"

#include <iostream>
#include <iomanip>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace BotUnitTest
{
	TEST_CLASS(BotUnitTest)
	{
	public:
		
		TEST_METHOD(TestMethod1)
		{
			Field field(20, 20);
			mt19937 mt;
			Assert::IsTrue(uct(&field, &mt, 100000) > -1, L"ALL OK");
		}

		TEST_METHOD(TestMethod2)
		{
			Field field(20, 20);
			field.MakeMove(field.to_pos(10, 10));
			mt19937 mt;
			Assert::IsTrue(uct(&field, &mt, 100000) > -1, L"ALL OK");
		}

		TEST_METHOD(TestMethod3)
		{
			Field field(20, 20);
			for (int i = 0; i < field.width; i++)
				for (int j = 0; j < field.height; j++)
					field.MakeMove(field.to_pos(i, j));
			ostringstream str;
			for (int i = 0; i < field.height; i++)
			{
				for (int j = 0; j < field.width; j++)
					str << setw(2) << (int)field.field[field.to_pos(j, i)] << " ";
				str << "\n";
			}
			Logger::WriteMessage(str.str().c_str());
		}

		TEST_METHOD(BotShouldCapture)
		{
			Field field(20, 20);
			field.field[field.to_pos(11, 9)] = 3;
			field.field[field.to_pos(11, 11)] = 3;
			field.field[field.to_pos(10, 10)] = 3;
			field.MakeMove(field.to_pos(11, 10));
			mt19937 mt;
			Move move = uct(&field, &mt, 1000);
			field.MakeMove(move);
			ostringstream str;
			for (int i = 0; i < field.height; i++)
			{
				for (int j = 0; j < field.width; j++)
					str << setw(2) << (int)field.field[field.to_pos(j, i)] << " ";
				str << "\n";
			}
			str << field.GetScore(0);
			Logger::WriteMessage(str.str().c_str());
			Assert::IsTrue(move == field.to_pos(12, 10), L"ALL OK");
		}

		TEST_METHOD(BotShouldCapture2)
		{
			Field field(20, 20, 1);
			field.field[field.to_pos(11, 9)] = 2;
			field.field[field.to_pos(11, 11)] = 2;
			field.field[field.to_pos(10, 10)] = 2;
			field.MakeMove(field.to_pos(11, 10));
			mt19937 mt;
			Move move = uct(&field, &mt, 200);
			field.MakeMove(move);
			ostringstream str;
			for (int i = 0; i < field.height; i++)
			{
				for (int j = 0; j < field.width; j++)
					str << setw(2) << (int)field.field[field.to_pos(j, i)] << " ";
				str << "\n";
			}
			str << field.GetScore(0);
			Logger::WriteMessage(str.str().c_str());
			Assert::IsTrue(move == field.to_pos(12, 10), L"ALL OK");
		}

		TEST_METHOD(BotShouldCaptureSmall1)
		{
			Field field(4, 4);
			field.field[field.to_pos(1, 0)] = 3;
			field.field[field.to_pos(0, 1)] = 3;
			field.field[field.to_pos(2, 1)] = 3;
			field.MakeMove(field.to_pos(1, 1));
			mt19937 mt;
			Move move = uct(&field, &mt, 100);
			field.MakeMove(move);
			ostringstream str;
			for (int i = 0; i < field.height; i++)
			{
				for (int j = 0; j < field.width; j++)
					str << setw(2) << (int)field.field[field.to_pos(j, i)] << " ";
				str << "\n";
			}
			str << (int)field.player << "\n";
			str << field.GetScore(0);
			Logger::WriteMessage(str.str().c_str());
			Assert::IsTrue(move == field.to_pos(1, 2), L"ALL OK");
		}

		TEST_METHOD(BotShouldCaptureSmall2)
		{
			Field field(4, 4, 1);
			field.field[field.to_pos(1, 0)] = 2;
			field.field[field.to_pos(0, 1)] = 2;
			field.field[field.to_pos(2, 1)] = 2;
			field.MakeMove(field.to_pos(1, 1));
			mt19937 mt;
			Move move = uct(&field, &mt, 50);
			field.MakeMove(move);
			ostringstream str;
			for (int i = 0; i < field.height; i++)
			{
				for (int j = 0; j < field.width; j++)
					str << setw(2) << (int)field.field[field.to_pos(j, i)] << " ";
				str << "\n";
			}
			str << field.GetScore(0);
			Logger::WriteMessage(str.str().c_str());
			Assert::IsTrue(move == field.to_pos(1, 2), L"ALL OK");
		}

		TEST_METHOD(ScoreTest)
		{
			Field field(20, 20);
			field.field[field.to_pos(11, 9)] = 3;
			field.field[field.to_pos(11, 11)] = 3;
			field.field[field.to_pos(10, 10)] = 3;
			field.MakeMove(field.to_pos(11, 10));
			field.MakeMove(field.to_pos(12, 10));
			Logger::WriteMessage(to_string(field.score).c_str());
			//mt19937 mt;
			//Move move = uct(&field, &mt, 10000);
			//field.MakeMove(move);
			//ostringstream str;
			//for (int i = 0; i < field.height; i++)
			//{
			//	for (int j = 0; j < field.width; j++)
			//		str << setw(2) << (int)field.field[field.to_pos(j, i)] << " ";
			//	str << "\n";
			//}
			//Logger::WriteMessage(str.str().c_str());
			Assert::AreEqual(1, field.GetScore(1), L"WRONG SCORE");
		}

		TEST_METHOD(FieldTest)
		{
			Field field(20, 20);
			field.MakeMove(field.to_pos(10, 10));
			field.MakeMove(field.to_pos(10, 11));
			field.MakeMove(field.to_pos(9, 11));
			field.MakeMove(field.to_pos(9, 12));
			field.MakeMove(field.to_pos(10, 12));
			field.MakeMove(field.to_pos(11, 12));
			field.MakeMove(field.to_pos(11, 11));
			field.Undo();
			ostringstream str;
			for (int i = 0; i < field.height; i++)
			{
				for (int j = 0; j < field.width; j++)
					str << setw(2) << (int)field.field[field.to_pos(j, i)] << " ";
				str << "\n";
			}
			Logger::WriteMessage(str.str().c_str());
			//Assert::AreEqual(str.str(), string(""));
		}
	};
}
