#pragma once
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vcruntime_string.h>

class ShareMemoryManage
{
public:
	void* operator new(size_t len)
	{
		void* ptr;
		cudaMallocManaged(&ptr,len);
		cudaDeviceSynchronize();
		return ptr;
	}

	void* operator new[](size_t len)
	{
		void* ptr;
		cudaMallocManaged(&ptr,len);
		cudaDeviceSynchronize();
		return ptr;
	}

		void operator delete(void* ptr)
	{
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}

	void operator delete[](void* ptr)
	{
		cudaDeviceSynchronize();
		cudaFree(ptr);
		cudaDeviceSynchronize();
	}
};