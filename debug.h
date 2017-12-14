#pragma once
#ifdef _WINDOWS
#include <windows.h>
#else
#include <string.h>
#include <stdio.h>
#include <time.h>
#define sprintf_s snprintf
#define _vsnprintf vsnprintf 
extern unsigned long GetTickCount();
#endif

extern void DBG_ASSERT(bool ast, const char * format, ...);
extern void DBG_PRINT(const char * format, ...);