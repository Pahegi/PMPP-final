#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <waveform_evolution.hpp>

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
	// auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(2 * device_wavefunction.size());
	// auto wave_out_span = cuda::std::span(wave_out.get(), 2 * device_wavefunction.size());
	std::uint64_t *wave_out_ptr;
	cudaMallocManaged(&wave_out_ptr, 2 * device_wavefunction.size() * sizeof(std::uint64_t));
	auto wave_out_span = cuda::std::span(wave_out_ptr, 2 * device_wavefunction.size());

	// Copy the input array to the output array
	size_t num_ed = device_wavefunction.size();
	cudaMemcpy(wave_out_span.data(), device_wavefunction.data(), device_wavefunction.size() * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

	// launch kernel
	int block_size;		// The launch configurator returned block size
	int min_grid_size;	// The minimum grid size needed to achieve the maximum occupancy for a full device launch
	int grid_size;		// The actual grid size needed, based on input size
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, evolve_operator_kernel, 0, 0);
	grid_size = (num_ed + block_size - 1) / block_size;	 // Round up according to array size
	evolve_operator_kernel<<<grid_size, block_size>>>(
		device_wavefunction.data(),
		wave_out_span.data(),
		num_ed,
		activation,
		deactivation);

	cudaDeviceSynchronize();

	// Sort and remove duplicates
	thrust::sort(wave_out_span.data(), wave_out_span.data() + 2 * num_ed);
	size_t *new_end = thrust::unique(wave_out_span.data(), wave_out_span.data() + 2 * num_ed);

	// Check if the first element is zero and adjust the return value
	size_t shift = (wave_out_span[0] == 0) ? 1 : 0;
	size_t num_ed_out = static_cast<std::size_t>(new_end - wave_out_span.data() - shift);
	return {pmpp::cuda_ptr<std::uint64_t[]>(wave_out_span.data() + shift), num_ed_out};
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations) {
	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result;
	auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(device_wavefunction.size());
	auto wave_out_span = cuda::std::span(wave_out.get(), device_wavefunction.size());

	std::copy_n(device_wavefunction.data(), device_wavefunction.size(), wave_out_span.data());

	for (size_t i = 0; i < activations.size(); i++) {
		result = evolve_operator(wave_out_span, activations[i], deactivations[i]);
		wave_out_span = cuda::std::span(result.first.get(), result.second);
	}

	return result;
}
