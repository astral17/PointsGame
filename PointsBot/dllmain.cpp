// dllmain.cpp : Определяет точку входа для приложения DLL.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "dllmain.h"

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

DLLEXPORT Field* Init(int width, int height, int seed)
{
    return new Field(width, height);
}
