#include <waveform_evolution.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <span>
#include <vector>

std::vector<std::uint64_t> evolve_operator_host(
	std::span<std::uint64_t> host_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
)
{
	using std::size;
	using std::data;

	auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_wavefunction));
	auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), size(host_wavefunction));
	std::copy_n(data(host_wavefunction), size(host_wavefunction), device_wavefunction.data());

	auto [result_wavefunction, result_size] = evolve_operator(device_wavefunction, activation, deactivation);

	std::vector<std::uint64_t> result(result_size);
	if(result_size)
		cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);
	return result;
}

std::vector<std::uint64_t> evolve_ansatz_host(
	std::span<std::uint64_t> host_wavefunction,
	std::span<std::uint64_t> host_activations,
	std::span<std::uint64_t> host_deactivations
)
{
	using std::size;
	using std::data;

	auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_wavefunction));
	auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), size(host_wavefunction));
	std::copy_n(data(host_wavefunction), size(host_wavefunction), device_wavefunction.data());

	auto device_activations_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_activations));
	auto device_activations = cuda::std::span(device_activations_ptr.get(), size(host_activations));
	std::copy_n(data(host_activations), size(host_activations), device_activations.data());

	auto device_deactivations_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(size(host_deactivations));
	auto device_deactivations = cuda::std::span(device_deactivations_ptr.get(), size(host_deactivations));
	std::copy_n(data(host_deactivations), size(host_deactivations), device_deactivations.data());

	auto [result_wavefunction, result_size] = evolve_ansatz(device_wavefunction, device_activations, device_deactivations);

	std::vector<std::uint64_t> result(result_size);
	if(result_size)
		cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);
	return result;
}

TEST_CASE("Trivial test", "[trivial]")
{
	REQUIRE(1 == 1);
}
