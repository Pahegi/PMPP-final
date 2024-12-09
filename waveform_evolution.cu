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

	printf("evolve_operator: starting with %u densities\n", device_wavefunction.size());
	printf("device_wavefunction 0:\n");
	for (size_t i = 0; i < device_wavefunction.size(); i++) {
		printf("%lu ", device_wavefunction[i]);
	}
	printf("\n");

	printf("evolve_operator: Creating Output Array\n");
	// auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(2 * device_wavefunction.size());
	// auto wave_out_span = cuda::std::span(wave_out.data(), 2 * device_wavefunction.size());
	std::uint64_t *wave_out_ptr;
	cudaMallocManaged(&wave_out_ptr, 2 * device_wavefunction.size() * sizeof(std::uint64_t));
	cudaMemset(wave_out_ptr, 0, 2 * device_wavefunction.size() * sizeof(std::uint64_t));
	auto wave_out_span = cuda::std::span(wave_out_ptr, 2 * device_wavefunction.size());

	size_t num_ed = device_wavefunction.size();

	// copy wavefunctions to output array
	printf("evolve_operator: Copying Wavefunctions to output Array\n");
	// Initialize wave_out_span with zeros
	// thrust::fill(wave_out_span.begin(), wave_out_span.end(), 0);
	printf("device_wavefunction 1:\n");
	for (size_t i = 0; i < device_wavefunction.size(); i++) {
		printf("%lu ", device_wavefunction[i]);
	}
	printf("\n");
	printf("device_wavefunction address: %p\n", device_wavefunction.data());
	printf("wave_out_span address: %p\n", wave_out_span.data());
	assert(device_wavefunction.data() != wave_out_span.data());
	assert((wave_out_span.data() + wave_out_span.size() <= device_wavefunction.data()) ||
		   (device_wavefunction.data() + device_wavefunction.size() <= wave_out_span.data()));

	cudaMemcpy(wave_out_span.data(), device_wavefunction.data(), device_wavefunction.size() * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
	printf("device_wavefunction 2:\n");
	for (size_t i = 0; i < device_wavefunction.size(); i++) {
		printf("%lu ", device_wavefunction[i]);
	}
	printf("\n");

	// optimize number of blocks and threads
	int num_blocks, num_threads;
	// calculate_blocks_and_threads_xdim(num_ed, &num_blocks, &num_threads);
	num_blocks = 32;
	num_threads = 32;
	// printf("evolve_operator: Blocks: %d, Threads per Block: %d\n", num_blocks, num_threads);

	// Check for any errors launching the kernel
	cudaError_t err_1 = cudaGetLastError();
	if (err_1 != cudaSuccess) {
		printf("evolve_operator: 1 Kernel launch failed: %s\n", cudaGetErrorString(err_1));
	}

	// launch kernel
	// Print the values
	printf("device_wavefunction 3:\n");
	for (size_t i = 0; i < device_wavefunction.size(); i++) {
		printf("%lu ", device_wavefunction[i]);
	}
	printf("\n");

	printf("wave_out_span:\n");
	for (size_t i = 0; i < wave_out_span.size(); ++i) {
		printf("%lu ", wave_out_span[i]);
	}
	printf("\n");
	printf("evolve_operator: Launching Kernel with %d blocks and %d threads, %lu densities, %d activation, %d deactivation\n", num_blocks, num_threads, num_ed, activation, deactivation);
	evolve_operator_kernel<<<num_blocks, num_threads>>>(
		device_wavefunction.data(),
		wave_out_span.data(),
		num_ed,
		activation,
		deactivation);

	// Check for any errors launching the kernel
	cudaError_t err_prev = cudaGetLastError();
	if (err_prev != cudaSuccess) {
		printf("evolve_operator: 2 Kernel launch failed: %s\n", cudaGetErrorString(err_prev));
	}

	// Wait for kernel to finish and check for errors
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("evolve_operator: 3 Kernel execution failed: %s\n", cudaGetErrorString(err));
	} else {
		printf("evolve_operator: 3 Kernel execution succeeded\n");
	}

	printf("evolve_operator: Kernel finished\n");

	// Sort and remove duplicates
	printf("evolve_operator: Sorting and Removing Duplicates\n");
	thrust::sort(wave_out_span.data(), wave_out_span.data() + 2 * num_ed);
	size_t *new_end = thrust::unique(wave_out_span.data(), wave_out_span.data() + 2 * num_ed);

	// Check if the first element is zero and adjust the return value
	size_t shift = (wave_out_span[0] == 0) ? 1 : 0;
	size_t num_ed_out = static_cast<std::size_t>(new_end - wave_out_span.data() - shift);
	printf("evolve_operator: Number of densities after evolution: %lu\n", num_ed_out);
	return {pmpp::cuda_ptr<std::uint64_t[]>(wave_out_span.data() + shift), num_ed_out};
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations) {
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result;
	auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(device_wavefunction.size());
	auto wave_out_span = cuda::std::span(wave_out.get(), device_wavefunction.size());

	printf("---------------------------------------------------------------------\n");

	printf("evolve_ansatz: Copying Wavefunctions to output Array\n");
	std::copy_n(device_wavefunction.data(), device_wavefunction.size(), wave_out_span.data());

	printf("evolve_ansatz: Initial number of operators: %lu\n", activations.size());
	printf("evolve_ansatz: Entries of wavefunction: ");
	for (size_t k = 0; k < device_wavefunction.size(); k++) {
		printf("%lu ", device_wavefunction.data()[k]);
	}
	printf("\n");

	for (size_t i = 0; i < activations.size(); i++) {
		printf("evolve_ansatz: Loop Number: %lu\n", i);

		result = evolve_operator(wave_out_span, activations[i], deactivations[i]);

		printf("evolve_ansatz: Result_size=%lu\n", result.second);
		printf("evolve_ansatz: Entries of wavefunction: ");
		for (size_t j = 0; j < result.second; j++) {
			printf("%lu ", result.first[j]);
		}
		printf("\n");

		wave_out_span = cuda::std::span(result.first.get(), result.second);
	}

	printf("evolve_ansatz: Returning number of densities after evolution: %lu\n", result.second);
	return result;
	// return {nullptr, 0};
}
