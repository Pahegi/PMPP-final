#include <stdio.h>

#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <span>
#include <unordered_set>
#include <vector>
#include <waveform_evolution.hpp>

#include "test_data_loader.hpp"

std::vector<std::uint64_t> evolve_operator_host(
	std::span<std::uint64_t const> host_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation) {
	using std::data;
	using std::size;

	auto t0 = std::chrono::system_clock::now();

	printf("evolve_operator_host: Allocating managed cuda array for %lu wavefunctions\n", host_wavefunction.size());
	auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(host_wavefunction.size());
	auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), host_wavefunction.size());

	printf("evolve_operator_host: Copying %lu host wavefunction to device\n", host_wavefunction.size());
	std::copy_n(data(host_wavefunction), size(host_wavefunction), device_wavefunction.data());

	printf("evolve_operator_host: Calling Evolve operator\n");
	auto [result_wavefunction, result_size] = evolve_operator(device_wavefunction, activation, deactivation);

	printf("evolve_operator_host: Copying %lu results back to host\n", result_size);
	std::vector<std::uint64_t> result(result_size);
	if (result_size)
		cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);

	auto t1 = std::chrono::system_clock::now();
	std::printf("Time: %ld µs\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
	return result;
}

std::vector<std::uint64_t> evolve_ansatz_host(
	std::span<std::uint64_t const> host_wavefunction,
	std::span<std::uint64_t const> host_activations,
	std::span<std::uint64_t const> host_deactivations) {
	using std::data;
	using std::size;

	printf("evolve_ansatz_host: Allocating managed cuda array for %lu wavefunctions\n", host_wavefunction.size());
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
	if (result_size)
		cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);

	return result;
}

TEST_CASE("Self test input data", "[self-test]") {
	test_data_loader loader("example_evolution.bin");

	auto electrons = loader.electrons();
	auto orbitals = loader.single_electron_density_count();
	auto activations = loader.activations();
	auto deactivations = loader.deactivations();

	REQUIRE(activations.size() == loader.ansatz_size());
	REQUIRE(deactivations.size() == loader.ansatz_size());

	auto orbital_mask = (orbitals < 64 ? std::uint64_t(1) << orbitals : 0) - 1;

	for (std::size_t i = 0, n = loader.ansatz_size(); i < n; ++i) {
		auto n_activations = std::popcount(activations[i]);
		auto n_deactivations = std::popcount(deactivations[i]);

		REQUIRE((activations[i] & deactivations[i]) == 0);

		REQUIRE((activations[i] & ~orbital_mask) == 0);
		REQUIRE((deactivations[i] & ~orbital_mask) == 0);

		REQUIRE(n_activations > 0);
		REQUIRE(n_activations <= 2);
		REQUIRE(n_activations == n_deactivations);
	}

	std::size_t step = 0;
	loader.for_each_step([&](
							 std::span<std::uint64_t const> wfn_in,
							 std::span<std::uint64_t const> wfn_out,
							 std::uint64_t activation,
							 std::uint64_t deactivation) {
		using std::begin;
		using std::end;

		REQUIRE(activation == activations[step]);
		REQUIRE(deactivation == deactivations[step]);

		auto wfn_in_set = std::unordered_set(begin(wfn_in), end(wfn_in));
		auto wfn_out_set = std::unordered_set(begin(wfn_out), end(wfn_out));
		REQUIRE(wfn_in_set.size() == wfn_in.size());
		REQUIRE(wfn_out_set.size() == wfn_out.size());

		REQUIRE(wfn_in.size() <= wfn_out.size());
		for (auto v : wfn_in)
			wfn_out_set.erase(v);
		REQUIRE(wfn_out_set.size() == wfn_out.size() - wfn_in.size());

		if (step == 0) {
			REQUIRE(std::all_of(begin(wfn_in), end(wfn_in), [&](std::uint64_t v) { return (v & ~orbital_mask) == 0; }));
			REQUIRE(std::all_of(begin(wfn_in), end(wfn_in), [&](std::uint64_t v) { return std::popcount(v) == electrons; }));
		}
		REQUIRE(std::all_of(begin(wfn_out), end(wfn_out), [&](std::uint64_t v) { return (v & ~orbital_mask) == 0; }));
		REQUIRE(std::all_of(begin(wfn_out), end(wfn_out), [&](std::uint64_t v) { return std::popcount(v) == electrons; }));

		++step;
	});

	REQUIRE(step == loader.ansatz_size());
}

TEST_CASE("Test evolve operator", "[simple]") {
	using std::begin;
	using std::end;

	test_data_loader loader("example_evolution.bin");
	loader.for_each_step([&](
							 std::span<std::uint64_t const> wfn_in,
							 std::span<std::uint64_t const> wfn_out,
							 std::uint64_t activation,
							 std::uint64_t deactivation) {
		// print wfn_in!!!
		printf("test_evolve_operator: elements of wfn_in: ");
		for (auto i : wfn_in) {
			std::cout << i << " ";
		}
		printf("\n");
		printf("test_evolve_operator: size of wfn_in: %lu\n", wfn_in.size());
		printf("test_evolve_operator: size of wfn_out: %lu\n", wfn_out.size());
		auto wfn_out_dut = evolve_operator_host(wfn_in, activation, deactivation);
		auto wfn_out_set = std::unordered_set(begin(wfn_out), end(wfn_out));
		auto wfn_out_dut_set = std::unordered_set(begin(wfn_out_dut), end(wfn_out_dut));
		REQUIRE(wfn_out_dut.size() == wfn_out_dut_set.size());
		REQUIRE(wfn_out_set == wfn_out_dut_set);
	});
}

// TEST_CASE("Custom Test evolve operator, "[simple]") {
// 	using std::begin;
// 	using std::end;

// 	std::vector<std::uint64_t> wfn_in = {1, 2, 3, 4, 5, 6, 7, 8};
// 	std::uint64_t activation = 1;
// 	std::uint64_t deactivation = 2;
// 	auto wfn_out = evolve_operator_host(wfn_in, activation, deactivation);
// 	printf("test_evolve_operator: size of wfn_out: %lu\n", wfn_out.size());
// 	printf("test_evolve_operator: elements of wfn_out: ");
// 	for (auto i : wfn_out) {
// 		std::cout << i << " ";
// 	}
// 	printf("\n");
// }

TEST_CASE("Test evolve ansatz", "[simple]") {
	using std::begin;
	using std::end;

	test_data_loader loader("example_evolution.bin");
	auto [wfn_in, wfn_out] = loader.first_and_last_wavefunction();
	printf("test_evolve_ansatz: size of wfn_in: %lu\n", wfn_in.size());
	auto wfn_out_dut = evolve_ansatz_host(wfn_in, loader.activations(), loader.deactivations());
	printf("test_evolve_ansatz: size of wfn_out: %lu\n", wfn_out_dut.size());
	printf("test_evolve_ansatz: elements of wfn_out: ");
	for (auto i : wfn_out_dut) {
		std::cout << i << " ";
	}
	printf("\n");
	auto wfn_out_set = std::unordered_set(begin(wfn_out), end(wfn_out));
	auto wfn_out_dut_set = std::unordered_set(begin(wfn_out_dut), end(wfn_out_dut));
	REQUIRE(wfn_out_dut.size() == wfn_out_dut_set.size());
	REQUIRE(wfn_out_set == wfn_out_dut_set);
}
