#include <stdio.h>

#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <sstream>
#include <string>
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

	auto device_wavefunction_ptr = pmpp::make_managed_cuda_array<std::uint64_t>(host_wavefunction.size());
	auto device_wavefunction = cuda::std::span(device_wavefunction_ptr.get(), host_wavefunction.size());

	std::copy_n(data(host_wavefunction), size(host_wavefunction), device_wavefunction.data());

	auto [result_wavefunction, result_size] = evolve_operator(device_wavefunction, activation, deactivation);

	std::vector<std::uint64_t> result(result_size);
	if (result_size)
		cudaMemcpy(data(result), result_wavefunction.get(), sizeof(std::uint64_t) * result_size, cudaMemcpyDefault);

	auto t1 = std::chrono::system_clock::now();
	std::printf("Operator time with %ld wavefunctions: %ld µs\n",
				size(host_wavefunction),
				std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
	return result;
}

std::vector<std::uint64_t> evolve_ansatz_host(
	std::span<std::uint64_t const> host_wavefunction,
	std::span<std::uint64_t const> host_activations,
	std::span<std::uint64_t const> host_deactivations) {
	using std::data;
	using std::size;

	auto t0 = std::chrono::system_clock::now();

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

	auto t1 = std::chrono::system_clock::now();
	std::printf("Ansatz time with %ld input wavefunctions, %ld operators and %ld output wavefunctions: %ld µs\n",
				size(host_wavefunction),
				size(host_activations),
				result_size,
				std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

	return result;
}

std::tuple<std::chrono::microseconds, size_t> run(int num_wavefunctions, int num_operators) {
	std::vector<std::uint64_t> host_wavefunction;
	std::vector<std::uint64_t> host_activations;
	std::vector<std::uint64_t> host_deactivations;
	std::uniform_int_distribution<std::uint64_t> bitdist(0, 63);
	std::default_random_engine gen;

	for (std::size_t i = 0; i < num_wavefunctions; ++i) {
		std::uint64_t wavefunction = 0;
		for (std::size_t i = 0; i < 2; ++i) {
			wavefunction |= 0x1lu << bitdist(gen);
		}
		host_wavefunction.push_back(wavefunction);
	}
	for (std::size_t i = 0; i < num_operators; ++i) {
		std::uint64_t act = 1;
		std::uint64_t dea = 1;
		while ((act & dea) != 0) {
			act = 0;
			dea = 0;
			act |= 0x1lu << bitdist(gen);
			dea |= 0x1lu << bitdist(gen);
			if ((bitdist(gen) & 0x1) == 1) {
				act |= 0x1lu << bitdist(gen);
				dea |= 0x1lu << bitdist(gen);
			}
		}
		host_activations.push_back(act);
		host_deactivations.push_back(dea);
	}

	auto t0 = std::chrono::system_clock::now();
	// run evolve_ansatz in seperate thread

	auto result = evolve_ansatz_host(host_wavefunction, host_activations, host_deactivations);
	auto t1 = std::chrono::system_clock::now();
	return {std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0), result.size()};
}

TEST_CASE("Test run with big data", "[simple]") {
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);
	std::stringstream ss;
	ss << "output_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") << ".csv";
	auto filename = ss.str();
	FILE *f = fopen(filename.c_str(), "a");
	if (f == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}

	// run single test to warm up the GPU
	run(10, 100);

	fprintf(f, "num_wavefunction, num_operator, time\n");
	for (int num_operator = 1; num_operator <= 100; num_operator += 10) {
		for (int num_wavefunction = 1; num_wavefunction <= 10000; num_wavefunction += 100) {
			auto [time, size] = run(num_wavefunction, num_operator);
			fprintf(f, "%d, %d, %ld, %d\n", num_wavefunction, num_operator, time.count(), size);
			fflush(f);
		}
	}

	fclose(f);
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
		auto wfn_out_dut = evolve_operator_host(wfn_in, activation, deactivation);
		auto wfn_out_set = std::unordered_set(begin(wfn_out), end(wfn_out));
		auto wfn_out_dut_set = std::unordered_set(begin(wfn_out_dut), end(wfn_out_dut));
		REQUIRE(wfn_out_dut.size() == wfn_out_dut_set.size());
		REQUIRE(wfn_out_set == wfn_out_dut_set);
	});
}

TEST_CASE("Test evolve ansatz", "[simple]") {
	using std::begin;
	using std::end;

	test_data_loader loader("example_evolution.bin");
	auto [wfn_in, wfn_out] = loader.first_and_last_wavefunction();
	auto wfn_out_dut = evolve_ansatz_host(wfn_in, loader.activations(), loader.deactivations());
	auto wfn_out_set = std::unordered_set(begin(wfn_out), end(wfn_out));
	auto wfn_out_dut_set = std::unordered_set(begin(wfn_out_dut), end(wfn_out_dut));
	REQUIRE(wfn_out_dut.size() == wfn_out_dut_set.size());
	REQUIRE(wfn_out_set == wfn_out_dut_set);
}
