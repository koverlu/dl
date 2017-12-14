#include <iostream>
#include <stdarg.h>
#include <assert.h>
#include "debug.h"

void DBG_ASSERT(bool ast, const char * format, ...)
{
	char buf[256];
	if (!ast)
	{
		va_list ap;
		va_start(ap, format);
		_vsnprintf(buf, sizeof(buf), format, ap);
		va_end(ap);
		std::cout << buf << std::endl;
#ifdef WIN_OS
		OutputDebugString(buf);
#endif
		assert(0);
	}
}

void DBG_PRINT(const char * format, ...)
{
	char buf[256];
	va_list ap;
	va_start(ap, format);
	_vsnprintf(buf, sizeof(buf), format, ap);
	va_end(ap);
	std::cout << buf << std::flush;
#ifdef WIN_OS
	OutputDebugString(buf);
#endif
}

#ifndef WIN_OS
unsigned long GetTickCount()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);  
}
#endif