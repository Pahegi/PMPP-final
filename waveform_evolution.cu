#include <waveform_evolution.hpp>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <algorithm>


// cuda kernel
__global__ void evolve_operator_kernel(
	std::uint64_t const* device_wavefunction,
	std::uint64_t* wave_out,
	std::size_t* size,
	std::uint64_t const activation,
	std::uint64_t const deactivation
)
{	

	// size_t output_elements = *size;
	auto op_ad = activation ^ deactivation;

	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx>=size[0]) return;

	auto wave = device_wavefunction[idx];
	auto check_deactivation = deactivation & wave; // TODO all bits or only one bit enough for deactivation?
	auto check_activation = activation & (~wave);
	uint64_t activate = check_activation == activation;
	uint64_t deactivate = check_deactivation == deactivation;
	auto op = (activate & deactivate) * op_ad;
	auto wave_new = op ^ wave;

			// insert new wavefunction, if wave_new != wave
	
	if (wave_new != wave) {
		// wave_out[idx] = wave_new;
		wave_out[idx+size[0]] = wave_new;

		// output_elements++;
	}
	else {
		wave_out[idx+size[0]] = 0;
	}

	//*size = output_elements;



	/*
	// only thread 0 (sequential implementation test), MASSIVELY PARALLEL!!!
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		// iterate over all wavefunctions
		for (std::size_t i = 0; i < *size; i++) {
			auto wave = device_wavefunction[i];
			auto check_deactivation = deactivation & wave; // TODO all bits or only one bit enough for deactivation?
			auto check_activation = activation & (~wave);
			uint64_t activate = check_activation == activation;
			uint64_t deactivate = check_deactivation == deactivation;
			auto op = (activate & deactivate) * op_ad;
			auto wave_new = op ^ wave;

			// insert new wavefunction, if wave_new != wave
			if (wave_new != wave) {
				wave_out[output_elements] = wave_new;
				output_elements++;
			}
		}
		*size = output_elements;

		printf("Block: %d Thread: %d Output Elements: %lu\n", threadIdx.x, blockIdx.x, output_elements);
	
	}
	*/
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{	
	// shift variable
	size_t shift = 0;

	// create output array double the size of input
	printf("evolve_operator: Creating Output Array\n");
	auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(2 * size(device_wavefunction));
	auto wave_out_span = cuda::std::span(wave_out.get(), 2 * size(device_wavefunction));
	
	printf("evolve_operator: Copying Wavefunctions to output Array\n");
	std::copy_n(device_wavefunction.data(), size(device_wavefunction), wave_out_span.data());

	// create size cuda pointer
	printf("evolve_operator: Creating Size Array\n");
	auto num_elements = pmpp::make_managed_cuda_array<std::size_t>(1);  // TODO make better
	num_elements[0] = size(device_wavefunction);
	auto num_blocks = 32;
	auto threads_per_block = (num_elements[0] / num_blocks) + (num_elements[0] % num_blocks > 0 ? 1 : 0); 
	// ((global_size / block_size) + (global_size % block_size > 0 ? 1 : 0));

	printf("evolve_operator: Launching Kernel\n");
	evolve_operator_kernel<<<num_blocks, threads_per_block>>>(
		device_wavefunction.data(),
		wave_out_span.data(),
		num_elements.get(),
		activation,
		deactivation
	);


	
	cudaDeviceSynchronize();

	thrust::sort(wave_out_span.data(), wave_out_span.data()+2*num_elements[0]);

	cudaDeviceSynchronize();

	size_t *new_end = thrust::unique(wave_out_span.data(), wave_out_span.data()+2*num_elements[0]);

	for(size_t i=0; i<num_elements[0]; i++){
		printf("%lu ", wave_out_span[i]);
	}
	printf("\n");

	cudaDeviceSynchronize();

	if(wave_out_span.data()[0]==0) {
		shift = 1;
	}


	printf("Num elements %lu\n", num_elements[0]);
	printf("evolve_operator: Returning output array\n");
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result {&(wave_out_span.data()[shift]), (new_end - wave_out_span.data())-shift};
	return result;


	// // create output array double the size of input
	// auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(2 * size(device_wavefunction));
	// auto wave_out_span = cuda::std::span(wave_out.get(), 2 * size(device_wavefunction));

	// // fill output with input waveforms
	// std::copy_n(device_wavefunction.data(), size(device_wavefunction), data(wave_out_span));


	// size_t output_elements = size(device_wavefunction);
	// auto op_ad = activation ^ deactivation;

	// // iterate over all wavefunctions
	// for (std::size_t i = 0; i < device_wavefunction.size(); i++) {
	// 	auto wave = device_wavefunction[i];
	// 	auto check_deactivation = deactivation & wave; // TODO all bits or only one bit enough for deactivation?
	// 	auto check_activation = activation & (~wave);
	// 	uint64_t activate = check_activation == activation;
	// 	uint64_t deactivate = check_deactivation == deactivation;
	// 	auto op = activate * deactivate * op_ad;
	// 	auto wave_new = op ^ wave;

	// 	// insert new wavefunction, if wave_new != wave
	// 	if (wave_new != wave) {
	// 		// check if wave_new already exists
	// 		bool exists = false;
	// 		for (std::size_t j = 0; j < output_elements; j++) {
	// 			if (wave_new == wave_out_span[j]) {
	// 				exists = true;
	// 				break;
	// 			}
	// 		}
	// 		if (!exists) {
	// 			wave_out_span[output_elements] = wave_new;
	// 			output_elements++;
	// 		}
	// 	}
	// }

	// cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result {wave_out_span.data(), output_elements};
	// return result;
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations
)
{
	/* TODO */
	return {nullptr, 0};
}
