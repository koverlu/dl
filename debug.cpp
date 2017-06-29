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
		OutputDebugString(buf);
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
	OutputDebugString(buf);
}