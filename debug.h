#pragma once
#ifdef WIN_OS
#include <windows.h>
#else
#include <time.h>
#define sprintf_s snprintf
#define _vsnprintf vsnprintf 
extern unsigned long GetTickCount();
#endif

extern void DBG_ASSERT(bool ast, const char * format, ...);
extern void DBG_PRINT(const char * format, ...);