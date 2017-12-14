#This file is to help build a standalone executable and static library
#for rgrsn.  It should work for both Windows and Linux.

cmake_minimum_required(VERSION 2.6)
project(lstm)

# build library
file(GLOB src
     *.cpp)
	 
add_executable(lstm ${src})
set_target_properties(lstm PROPERTIES OUTPUT_NAME "lstm")

