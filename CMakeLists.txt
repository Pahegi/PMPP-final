# tested with module load cmake/3.26.1 gcc/13.1.0 cuda/12.5

cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(pmpp_final CXX CUDA)

# cmake standard build options
option(BUILD_TESTING "Control whether the CTest module invokes enable_testing()" OFF)
option(BUILD_SHARED_LIBS "Tell add_library() to default to SHARED libraries, instead of STATIC libraries, when called with no explicit library type" OFF)

foreach(lang CXX CUDA)
	set(CMAKE_${lang}_STANDARD 20)
	set(CMAKE_${lang}_STANDARD_REQUIRED ON)
	set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

if(MSVC)
	add_link_options(/NODEFAULTLIB:LIBCMT)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_library(
	pmpp_final
	waveform_evolution.cu
	include/waveform_evolution.hpp
	include/helpers.hpp
)

target_include_directories(pmpp_final PUBLIC include)

find_package(CUDAToolkit REQUIRED)
find_package(libcudacxx REQUIRED)
target_link_libraries(
	pmpp_final
	PUBLIC
	$<IF:$<STREQUAL:$<UPPER_CASE:${CMAKE_CUDA_RUNTIME_LIBRARY}>,"SHARED">,CUDA::cudart,CUDA::cudart_static>
	libcudacxx::libcudacxx
)

if(NOT BUILD_TESTING)
	return()
endif()

include(FetchContent)
FetchContent_Declare(
	Catch2
	GIT_SHALLOW ON
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG v3.7.1
	OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Catch2)
find_package(Catch2 3 REQUIRED)
include(CTest)
include(Catch)

add_executable(
	test_pmpp_final
	test_pmpp_final.cpp
)

target_link_libraries(
	test_pmpp_final
	PRIVATE
	pmpp_final
	Catch2::Catch2WithMain
)

catch_discover_tests(
	test_pmpp_final
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
