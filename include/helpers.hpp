#pragma once

#include <memory>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime_api.h>

namespace pmpp
{
	struct cuda_deleter
	{
		void operator() (void * p) const { cudaFree(p); }
	};

	template<typename T>
	using cuda_ptr = std::unique_ptr<T, cuda_deleter>;

	template<typename T>
	cuda_ptr<T[]> make_managed_cuda_array(
		std::size_t elements,
		unsigned flags = cudaMemAttachGlobal,
		cudaError_t * error = nullptr
	)
	{
		static_assert(std::is_trivially_destructible_v<T>, "T must be trivially destructible");
		void * pointer = nullptr;
		auto err = cudaMallocManaged(&pointer, sizeof(T) * elements, flags);
		if(error) *error = err;
		if(!pointer) throw std::bad_alloc();
		return cuda_ptr<T[]>(static_cast<T *>(pointer));
	}

	template<typename T>
	cuda_ptr<T[]> make_cuda_array(
		std::size_t elements,
		cudaError_t * error = nullptr
	)
	{
		static_assert(std::is_trivially_destructible_v<T>, "T must be trivially destructible");
		void * pointer = nullptr;
		auto err = cudaMalloc(&pointer, sizeof(T) * elements);
		if(error) *error = err;
		if(!pointer) throw std::bad_alloc();
		return cuda_ptr<T[]>(static_cast<T *>(pointer));
	}
}
