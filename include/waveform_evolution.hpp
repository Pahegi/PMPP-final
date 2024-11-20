#pragma once

#include <helpers.hpp>
#include <cuda/std/utility>
#include <cuda/std/span>

/// @brief Evolve a wavefunction using a single operator
/// @param device_wavefunction Current wavefunction (stored in device or managed memory)
/// @param activation Activation part of the operator
/// @param deactivation Deactivation part of the operator
/// @return The resulting wavefunction (stored in device or managed memory) and the size of the resulting wavefunction
cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_operator(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	std::uint64_t activation, std::uint64_t deactivation
);

/// @brief Evolve a wavefunction using a full ansatz
/// @param device_wavefunction Initial wavefunction (stored in device or managed memory)
/// @param activation Activation parts of the operators (stored in device or managed memory)
/// @param deactivation Deactivation parts of the operators (stored in device or managed memory; same size as activation)
/// @return The final wavefunction (stored in device or managed memory) and the size of the final wavefunction
cuda::std::pair<pmpp::cuda_ptr<std::uint64_t[]>, std::size_t> evolve_ansatz(
	cuda::std::span<std::uint64_t const> device_wavefunction,
	cuda::std::span<std::uint64_t const> activations,
	cuda::std::span<std::uint64_t const> deactivations
);
