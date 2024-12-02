#include <waveform_evolution.hpp>
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
	size_t output_elements = *size;
	auto op_ad = activation ^ deactivation;

	// only thread 0 (sequential implementation test), MASSIVELY PARALLEL!!!
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		// iterate over all wavefunctions
		for (std::size_t i = 0; i < *size; i++) {
			auto wave = device_wavefunction[i];
			auto check_deactivation = deactivation & wave; // TODO all bits or only one bit enough for deactivation?
			auto check_activation = activation & (~wave);
			uint64_t activate = check_activation == activation;
			uint64_t deactivate = check_deactivation == deactivation;
			auto op = activate * deactivate * op_ad;
			auto wave_new = op ^ wave;

			// insert new wavefunction, if wave_new != wave
			if (wave_new != wave) {
				wave_out[output_elements] = wave_new;
				output_elements++;
			}
		}
		*size = output_elements;

	}
}

cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{
	// // create output array double the size of input
	// auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(2 * size(device_wavefunction));
	// auto wave_out_span = cuda::std::span(wave_out.get(), 2 * size(device_wavefunction));
	// std::copy_n(device_wavefunction.data(), size(device_wavefunction), wave_out_span.data());

	// // create size cuda pointer
	// auto num_elements = pmpp::make_managed_cuda_array<std::size_t>(1);  // TODO make better
	// num_elements[0] = size(device_wavefunction);

	// evolve_operator_kernel<<<1, 1>>>(
	// 	data(device_wavefunction),
	// 	wave_out_span.data(),
	// 	num_elements.get(),
	// 	activation,
	// 	deactivation
	// );

	// cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result {wave_out_span.data(), num_elements[0]};
	// return result;


	// create output array double the size of input
	auto wave_out = pmpp::make_managed_cuda_array<std::uint64_t>(2 * size(device_wavefunction));
	auto wave_out_span = cuda::std::span(wave_out.get(), 2 * size(device_wavefunction));

	// fill output with input waveforms
	std::copy_n(device_wavefunction.data(), size(device_wavefunction), data(wave_out_span));


	size_t output_elements = size(device_wavefunction);
	auto op_ad = activation ^ deactivation;

	// iterate over all wavefunctions
	for (std::size_t i = 0; i < device_wavefunction.size(); i++) {
		auto wave = device_wavefunction[i];
		auto check_deactivation = deactivation & wave; // TODO all bits or only one bit enough for deactivation?
		auto check_activation = activation & (~wave);
		uint64_t activate = check_activation == activation;
		uint64_t deactivate = check_deactivation == deactivation;
		auto op = activate * deactivate * op_ad;
		auto wave_new = op ^ wave;

		// insert new wavefunction, if wave_new != wave
		if (wave_new != wave) {
			// check if wave_new already exists
			bool exists = false;
			for (std::size_t j = 0; j < output_elements; j++) {
				if (wave_new == wave_out_span[j]) {
					exists = true;
					break;
				}
			}
			if (!exists) {
				wave_out_span[output_elements] = wave_new;
				output_elements++;
			}
		}
	}

	cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> result {wave_out_span.data(), output_elements};
	return result;
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
