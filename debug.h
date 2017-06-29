#pragma once
#include <windows.h>

extern void DBG_ASSERT(bool ast, const char * format, ...);
extern void DBG_PRINT(const char * format, ...);