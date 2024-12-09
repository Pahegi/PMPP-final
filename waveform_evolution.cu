#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <waveform_evolution.hpp>

void calculate_blocks_and_threads_xdim(int n, int *num_blocks, int *num_threads) {
	int max_threads_per_block, max_grid_dim_x;

	// Get device properties
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);	// Assume using device 0
	max_threads_per_block = prop.maxThreadsPerBlock;
	max_grid_dim_x = prop.maxGridSize[0];  // Maximum number of blocks in the x-dimension

	// Set num_threads based on a typical value or maximum supported by the device
	*num_threads = (max_threads_per_block < 512) ? max_threads_per_block : 512;

	// Calculate num_blocks
	*num_blocks = (n + *num_threads - 1) / *num_threads;  // ceil(n / num_threads)

	// Ensure num_blocks does not exceed the maximum grid size in x-dimension
	if (*num_blocks > max_grid_dim_x) {
		printf("Error: Too many blocks required (%d), exceeds device capability (%d).\n", *num_blocks, max_grid_dim_x);
		*num_blocks = max_grid_dim_x;  // Cap at the maximum grid size
	}
}

__global__ void evolve_operator_kernel(
	std::uint64_t const *device_wavefunction,
	std::uint64_t *wave_out,
	std::size_t num_ed,
	std::uint64_t const activation,
	std::uint64_t const deactivation) {
	// only calculate if idx is within bounds
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_ed) return;

	// calculate new wavefunction
	auto op_ad = activation ^ deactivation;
	auto wave = device_wavefunction[idx];
	auto check_deactivation = deactivation & wave;
	auto check_activation = activation & (~wave);
	auto op = ((check_activation == activation) & (check_deactivation == deactivation)) * op_ad;
	auto wave_new = op ^ wave;

	// insert new wavefunction, if wave_new != wave
	wave_out[idx + num_ed] = (wave_new != wave) ? wave_new : 0;
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation) {
	// create output array double the size of input

	printf("evolce_operator: starting with %u densities\n", device_wavefunction.size());
	printf("Entries in the passing array:\n");
	for (size_t k = 0; k < device_wavefunction.size(); k++) {
		printf("%lu ", device_wavefunction[k]);
	}
	printf("\n");

	printf("evolve_operator: Creating Output Array\n");
	auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(2 * device_wavefunction.size());
	auto wave_out_span = cuda::std::span(wave_out.get(), 2 * device_wavefunction.size());
	size_t num_ed = device_wavefunction.size();

	// copy wavefunctions to output array
	printf("evolve_operator: Copying Wavefunctions to output Array\n");
	std::copy_n(device_wavefunction.data(), device_wavefunction.size(), wave_out_span.data());

	// optimize number of blocks and threads
	int num_blocks, num_threads;
	calculate_blocks_and_threads_xdim(num_ed, &num_blocks, &num_threads);
	printf("evolve_operator: Blocks: %d, Threads per Block: %d\n", num_blocks, num_threads);

	// launch kernel
	printf("evolve_operator: Launching Kernel\n");
	evolve_operator_kernel<<<num_blocks, num_threads>>>(
		device_wavefunction.data(),
		wave_out_span.data(),
		num_ed,
		activation,
		deactivation);
	cudaDeviceSynchronize();

	printf("evolve_operator: Kernel finished\n");

	// Sort and remove duplicates
	printf("evolve_operator: Sorting and Removing Duplicates\n");
	thrust::sort(wave_out_span.data(), wave_out_span.data() + 2 * num_ed);
	size_t *new_end = thrust::unique(wave_out_span.data(), wave_out_span.data() + 2 * num_ed);

	// Check if the first element is zero and adjust the return value
	size_t shift = (wave_out_span[0] == 0) ? 1 : 0;
	size_t num_ed_out = static_cast<std::size_t>(new_end - wave_out_span.data() - shift);
	printf("evolve_operator: Number of densities after evolution: %lu\n", num_ed_out);
	return {pmpp::cuda_ptr<std::uint64_t[]>(wave_out.get() + shift), num_ed_out};
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations) {
	/* TODO */
	size_t iterations = activations.size();
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result;
	auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(device_wavefunction.size());
	auto wave_out_span = cuda::std::span(wave_out.get(), device_wavefunction.size());

	printf("---------------------------------------------------------------------\n");

	printf("evolve_ansatz: Copying Wavefunctions to output Array\n");
	std::copy_n(device_wavefunction.data(), device_wavefunction.size(), wave_out_span.data());
	// uint64_t activation

	printf("evolve_ansatz: Initial number of operators: %lu\n", iterations);
	printf("evolve_ansatz: Entries of wavefunction:\n");

	for (size_t k = 0; k < device_wavefunction.size(); k++) {
		printf("%lu ", device_wavefunction.data()[k]);
	}
	printf("\n");
	for (size_t i = 0; i < iterations; i++) {
		printf("evolve_ansatz: Loop Number: %lu\n", i);
		/*
		if (i == 0){
			auto [result_wavefunct, result_size] = evolve_operator(wave_out_span, activations[i], deactivations[i]);
		} else {
			auto [result_wavefunct, result_size] = evolve_operator(wave_out_span, activations[i], deactivations[i]);
		}
		*/
		result = evolve_operator(wave_out_span, activations[i], deactivations[i]);

		printf("evolve_ansatz: Result_size=%lu\n", result.second);
		printf("evolve_ansatz: Entries of wavefunction:\n");
		for (size_t j = 0; j < result.second; j++) {
			printf("%lu ", result.first[j]);
		}
		printf("\n");

		wave_out_span = cuda::std::span(result.first.get(), result.second);
	}

	printf("evolve_ansatz: Entries of wavefunction:\n");
	for (size_t j = 0; j < result.second; j++) {
		printf("%lu ", result.first[j]);
	}
	printf("\n");
	printf("evolve_ansatz: Returning number of densities after evolution: %lu\n", result.second);
	return result;
	// return {nullptr, 0};
}
