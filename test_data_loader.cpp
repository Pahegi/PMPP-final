#include "test_data_loader.hpp"

#include <cstdio>

#include <stdexcept>

namespace
{
	struct file_closer
	{
		void operator() (std::FILE *file) const { std::fclose(file); }
	};

	using file_ptr = std::unique_ptr<std::FILE, file_closer>;
}

struct test_data_loader::impl
{
	explicit impl(char const * file_name)
		: file(std::fopen(file_name, "rb"))
	{
		auto f = file.get();
		if(!f)
			throw std::runtime_error("failed to open test data file");

		if(std::fread(&electrons, sizeof(electrons), 1, f) != 1)
			throw std::runtime_error("failed to read electron count");
		if(std::fread(&single_electron_density_count, sizeof(single_electron_density_count), 1, f) != 1)
			throw std::runtime_error("failed to read single electron density count");

		std::size_t ansatz_size_ = {};
		if(std::fread(&ansatz_size_, sizeof(ansatz_size_), 1, f) != 1)
			throw std::runtime_error("failed to read ansatz size");

		activations.resize(ansatz_size_); deactivations.resize(ansatz_size_);
		if(std::fread(deactivations.data(), sizeof(deactivations[0]), deactivations.size(), f) != deactivations.size())
			throw std::runtime_error("failed to read deactivations");
		if(std::fread(activations.data(), sizeof(activations[0]), activations.size(), f) != activations.size())
			throw std::runtime_error("failed to read activations");

		header_end = std::ftell(f);
		if(header_end < 0)
			throw std::runtime_error("failed to determine header end");
	}

	file_ptr file;
	std::size_t electrons = 0, single_electron_density_count = 0;
	std::vector<std::uint64_t> activations, deactivations;
	long header_end = 0;
};

test_data_loader::test_data_loader(char const * file_name)
	: pimpl(std::make_unique<test_data_loader::impl>(file_name))
{}
test_data_loader::~test_data_loader() = default;

std::size_t test_data_loader::electrons() const noexcept { return pimpl->electrons; }
std::size_t test_data_loader::single_electron_density_count() const noexcept { return pimpl->single_electron_density_count; }
std::size_t test_data_loader::ansatz_size() const noexcept { return pimpl->activations.size(); }
std::span<std::uint64_t const> test_data_loader::activations() const noexcept { return pimpl->activations; }
std::span<std::uint64_t const> test_data_loader::deactivations() const noexcept { return pimpl->deactivations; }

std::vector<std::uint64_t> test_data_loader::first_wavefunction()
{
    auto f = pimpl->file.get();

        if(std::fseek(f, pimpl->header_end, SEEK_SET))
                throw std::runtime_error("failed to seek to header end");

    std::size_t wavefunction_size = {};
    if(std::fread(&wavefunction_size, sizeof(wavefunction_size), 1, f) != 1)
        throw std::runtime_error("failed to read wavefunction size");

    std::vector<std::uint64_t> first(wavefunction_size);
    if(std::fread(first.data(), sizeof(first[0]), first.size(), f) != first.size())
        throw std::runtime_error("failed to read wavefunction");

    return first;
}

std::pair<std::vector<std::uint64_t>, std::vector<std::uint64_t>> test_data_loader::first_and_last_wavefunction()
{
	auto f = pimpl->file.get();

	if(std::fseek(f, pimpl->header_end, SEEK_SET))
		throw std::runtime_error("failed to seek to header end");

	std::vector<std::uint64_t> first, current;
	for(std::size_t i = 0, n = ansatz_size() + 1; i < n; ++i)
	{
		std::size_t wavefunction_size = {};
		if(std::fread(&wavefunction_size, sizeof(wavefunction_size), 1, f) != 1)
			throw std::runtime_error("failed to read wavefunction size");

		current.resize(wavefunction_size);
		if(std::fread(current.data(), sizeof(current[0]), current.size(), f) != current.size())
			throw std::runtime_error("failed to read wavefunction");

		if(i == 0)
			first = std::move(current);
	}

	return {std::move(first), std::move(current)};
}

void test_data_loader::for_each_step(std::function<void (std::span<std::uint64_t const>, std::span<std::uint64_t const>, std::uint64_t, std::uint64_t)> functor)
{
	auto f = pimpl->file.get();

	if(std::fseek(f, pimpl->header_end, SEEK_SET))
		throw std::runtime_error("failed to seek to header end");

	std::vector<std::uint64_t> previous, current;
	for(std::size_t i = 0, n = ansatz_size() + 1; i < n; ++i)
	{
		std::size_t wavefunction_size = {};
		if(std::fread(&wavefunction_size, sizeof(wavefunction_size), 1, f) != 1)
			throw std::runtime_error("failed to read wavefunction size");

		current.resize(wavefunction_size);
		if(std::fread(current.data(), sizeof(current[0]), current.size(), f) != current.size())
			throw std::runtime_error("failed to read wavefunction");

		if(i >= 1)
			functor(previous, current, activations()[i - 1], deactivations()[i - 1]);
		std::swap(previous, current);
	}
}
