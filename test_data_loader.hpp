#pragma once

#include <cstdint>

#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

struct test_data_loader
{
	explicit test_data_loader(char const * file_name);
	~test_data_loader();

	std::size_t electrons() const noexcept;
	std::size_t single_electron_density_count() const noexcept;
	std::size_t ansatz_size() const noexcept;
	std::span<std::uint64_t const> activations() const noexcept;
	std::span<std::uint64_t const> deactivations() const noexcept;

	std::pair<std::vector<std::uint64_t>, std::vector<std::uint64_t>> first_and_last_wavefunction();
	void for_each_step(std::function<void (std::span<std::uint64_t const>, std::span<std::uint64_t const>, std::uint64_t, std::uint64_t)> functor);

private:
	struct impl;
	std::unique_ptr<impl> pimpl;
};
