// Device matrix contracts: slice assignment against an independent indexing
// oracle on CUDA and Metal; reseeding against independent cuRAND references on
// CUDA, and reseed/repeat behaviour on Metal (MTGP32 has no host reference).
#include "../cpp/juzhen.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(CUDA) && !defined(APPLE_SILICON)
int compute() {
    LOG_INFO("device matrix contract tests require CUDA or Metal; skipping");
    return 77;
}
#else
namespace {
#ifdef CUDA
using Device = Matrix<CUDAfloat>;
constexpr const char* backend = "CUDA";
void device_sync() { CudaErrorCheck(cudaDeviceSynchronize()); }
#else
using Device = Matrix<MPSfloat>;
constexpr const char* backend = "Metal";
void device_sync() { mpsSynchronize(); }
#endif

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// The oracle is row-major host storage, independent of Juzhen slice/transpose.
struct Grid {
    size_t rows, columns;
    std::vector<float> values;
    Grid(size_t r, size_t c) : rows(r), columns(c), values(r * c) {}
    float& at(size_t r, size_t c) { return values[r * columns + c]; }
    float at(size_t r, size_t c) const { return values[r * columns + c]; }
};

Grid fixture(size_t rows, size_t columns, float base) {
    Grid result(rows, columns);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < columns; ++c)
            result.at(r, c) = base + float((columns + 1) * r + c);
    return result;
}

Grid transposed(const Grid& input) {
    Grid result(input.columns, input.rows);
    for (size_t r = 0; r < input.rows; ++r)
        for (size_t c = 0; c < input.columns; ++c)
            result.at(c, r) = input.at(r, c);
    return result;
}

Device upload(const Grid& logical, bool transpose) {
    Matrix<float> physical("contract fixture", transpose ? logical.columns : logical.rows,
                           transpose ? logical.rows : logical.columns);
    for (size_t r = 0; r < logical.rows; ++r)
        for (size_t c = 0; c < logical.columns; ++c)
            physical.elem(transpose ? c : r, transpose ? r : c) = logical.at(r, c);
    Device result(physical);
    if (transpose) return result.T();
    return result;
}

void exact(const Device& actual, const Grid& expected, const std::string& message) {
    require(actual.num_row() == expected.rows && actual.num_col() == expected.columns,
            message + " shape");
    const auto host = actual.to_host();
    for (size_t r = 0; r < expected.rows; ++r)
        for (size_t c = 0; c < expected.columns; ++c)
            if (host.elem(r, c) != expected.at(r, c))
                throw std::runtime_error(message + " at (" + std::to_string(r) + "," +
                                         std::to_string(c) + "): actual=" +
                                         std::to_string(host.elem(r, c)) + " expected=" +
                                         std::to_string(expected.at(r, c)));
}

void assign_reference(Grid& destination, const Grid& source, size_t row, size_t column) {
    for (size_t r = 0; r < source.rows; ++r)
        for (size_t c = 0; c < source.columns; ++c)
            destination.at(row + r, column + c) = source.at(r, c);
}

enum class Assignment { Slice, Rows, Columns };

void assignment_case(bool destination_transpose, bool source_transpose, Assignment operation,
                     size_t rows, size_t columns, size_t r0, size_t r1, size_t c0, size_t c1) {
    auto expected = fixture(rows, columns, -2000);
    const auto source_grid = fixture(r1 - r0, c1 - c0, 3000);
    auto destination = upload(expected, destination_transpose);
    auto source = upload(source_grid, source_transpose);
    require(bool(destination.get_transpose()) == destination_transpose &&
            bool(source.get_transpose()) == source_transpose, "fixture transpose flags");
    const std::string tag = "slice destination=" + std::to_string(destination_transpose) +
                            " source=" + std::to_string(source_transpose) +
                            " operation=" + std::to_string(int(operation));
    // Check that fixture construction itself agrees with the independent oracle.
    exact(destination, expected, tag + " initial destination");
    exact(source, source_grid, tag + " initial source");
    if (operation == Assignment::Rows) destination.rows(r0, r1, source);
    else if (operation == Assignment::Columns) destination.columns(c0, c1, source);
    else destination.slice(r0, r1, c0, c1, source);
    assign_reference(expected, source_grid, r0, c0);
    exact(destination, expected, tag + " written and untouched cells");
    exact(source, source_grid, tag + " source preserved");
    require(bool(destination.get_transpose()) == destination_transpose, tag + " layout preserved");
}

template<class Function>
void rejects_without_writing(Device& destination, const Grid& before,
                             Function operation, const std::string& message) {
    bool rejected = false;
    try { operation(); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, message + " must reject");
    device_sync();
    exact(destination, before, message + " destination preserved");
}

void invalid_and_empty_cases(bool transpose) {
    const auto before = fixture(4, 6, -1000);
    auto destination = upload(before, transpose);
    auto patch = upload(fixture(2, 3, 2000), !transpose);
    auto wrong_shape = upload(fixture(3, 2, 2000), transpose);
    rejects_without_writing(destination, before, [&] {
        destination.slice(1, 3, 2, 5, wrong_shape);
    }, "same element count but different source shape");
    auto too_small = upload(fixture(1, 1, 2000), !transpose);
    rejects_without_writing(destination, before, [&] {
        destination.slice(1, 3, 2, 5, too_small);
    }, "source too small");
    rejects_without_writing(destination, before, [&] {
        destination.slice(3, 5, 2, 5, patch);
    }, "row endpoint out of bounds");
    rejects_without_writing(destination, before, [&] {
        destination.slice(1, 3, 4, 7, patch);
    }, "column endpoint out of bounds");
    rejects_without_writing(destination, before, [&] {
        destination.slice(3, 1, 2, 5, patch);
    }, "reversed row range");
    rejects_without_writing(destination, before, [&] {
        destination.slice(1, 3, 5, 2, patch);
    }, "reversed column range");
    const size_t maximum = std::numeric_limits<size_t>::max();
    rejects_without_writing(destination, before, [&] {
        destination.slice(0, maximum, 0, maximum, patch);
    }, "maximum size_t endpoints rejected before size arithmetic");
    rejects_without_writing(destination, before, [&] {
        destination.slice(maximum, 3, 2, 5, patch);
    }, "maximum size_t reversed row start");
    rejects_without_writing(destination, before, [&] {
        destination.slice(1, 3, maximum, 5, patch);
    }, "maximum size_t reversed column start");
    rejects_without_writing(destination, before, [&] {
        destination.rows(1, 3, patch);
    }, "rows wrapper rejects source width");
    rejects_without_writing(destination, before, [&] {
        destination.columns(2, 5, patch);
    }, "columns wrapper rejects source height");
    rejects_without_writing(destination, before, [&] {
        destination.rows(3, 5, patch);
    }, "rows wrapper rejects bounds");
    rejects_without_writing(destination, before, [&] {
        destination.columns(5, 8, patch);
    }, "columns wrapper rejects bounds");

    for (bool source_transpose : {false, true}) {
        auto empty_rows = upload(Grid(0, 3), source_transpose);
        auto empty_columns = upload(Grid(2, 0), source_transpose);
        auto empty_both = upload(Grid(0, 0), source_transpose);
        destination.slice(4, 4, 2, 5, empty_rows);
        destination.slice(1, 3, 6, 6, empty_columns);
        destination.slice(4, 4, 6, 6, empty_both);
        auto full_width_empty = upload(Grid(0, 6), source_transpose);
        auto full_height_empty = upload(Grid(4, 0), source_transpose);
        destination.rows(4, 4, full_width_empty);
        destination.columns(6, 6, full_height_empty);
        device_sync();
        exact(destination, before, "valid empty ranges are no-ops");
        rejects_without_writing(destination, before, [&] {
            destination.slice(4, 4, 2, 5, empty_both);
        }, "empty range still checks source shape");
        rejects_without_writing(destination, before, [&] {
            destination.slice(5, 5, 2, 5, empty_rows);
        }, "empty row range outside bounds");
        rejects_without_writing(destination, before, [&] {
            destination.slice(1, 3, 7, 7, empty_columns);
        }, "empty column range outside bounds");
        rejects_without_writing(destination, before, [&] {
            destination.slice(maximum, maximum, 2, 5, empty_rows);
        }, "maximum size_t empty row range still rejects bounds");
        rejects_without_writing(destination, before, [&] {
            destination.slice(1, 3, maximum, maximum, empty_columns);
        }, "maximum size_t empty column range still rejects bounds");
    }
}

void alias_cases(bool transpose) {
    const auto initial = fixture(33, 33, 100);
    auto destination = upload(initial, transpose);
    destination.slice(0, 33, 0, 33, destination);
    exact(destination, initial, "direct self assignment");
    // T() returns shared storage, unlike the deep-copy constructor.
    auto same_layout = destination.T().T();
    require(same_layout.data() == destination.data(), "double transpose must alias");
    destination.slice(0, 33, 0, 33, same_layout);
    exact(destination, initial, "shared storage with same layout");
    auto opposite_layout = destination.T();
    require(opposite_layout.data() == destination.data(), "transpose must alias");
    destination.slice(0, 33, 0, 33, opposite_layout);
    // Reuse a same-sized pool allocation before any host copy synchronizes the
    // transpose. Its fill must follow the kernel that still reads the snapshot.
    auto reused_storage = Device::ones(33, 33);
    exact(destination, transposed(initial), "in-place transpose snapshots the source");
    Grid expected_ones(33, 33);
    for (float& value : expected_ones.values) value = 1.f;
    exact(reused_storage, expected_ones, "immediate pool reuse preserves both results");
    // Check the alias is still a live view of the updated storage.
    exact(opposite_layout, initial, "shared transpose view sees updated storage");
}

void slice_checks() {
    for (bool destination_transpose : {false, true}) {
        for (bool source_transpose : {false, true}) {
            assignment_case(destination_transpose, source_transpose, Assignment::Slice,
                            3, 5, 0, 3, 0, 5);
            assignment_case(destination_transpose, source_transpose, Assignment::Slice,
                            7, 9, 2, 5, 1, 6);
            assignment_case(destination_transpose, source_transpose, Assignment::Rows,
                            7, 9, 2, 5, 0, 9);
            assignment_case(destination_transpose, source_transpose, Assignment::Columns,
                            7, 9, 0, 7, 2, 5);
            assignment_case(destination_transpose, source_transpose, Assignment::Slice,
                            7, 9, 2, 3, 3, 6);
            assignment_case(destination_transpose, source_transpose, Assignment::Slice,
                            7, 9, 2, 5, 3, 4);
            // Cross block boundaries for every source/destination layout pair.
            assignment_case(destination_transpose, source_transpose, Assignment::Slice,
                            41, 59, 4, 37, 7, 54);
        }
        invalid_and_empty_cases(destination_transpose);
        alias_cases(destination_transpose);
    }
    // This non-square patch exceeds 4096 * 256 threads, so the tail requires
    // another grid-stride iteration while reading a transposed source.
    static_assert(size_t(1025) * 1027 > size_t(4096) * 256);
    assignment_case(false, true, Assignment::Slice, 1030, 1034, 2, 1027, 3, 1030);
    std::cout << "[PASS] " << backend << " slice assignment contracts\n";
}

Device draw(bool normal, size_t rows, size_t columns) {
    return normal ? Device::randn(rows, columns) : Device::rand(rows, columns);
}

void consume_both_distributions() {
    auto uniform = Device::rand(3, 5);
    auto normal = Device::randn(2, 8);
    device_sync();
}

Grid host_grid(const Device& m) {
    const auto host = m.to_host();
    Grid result(m.num_row(), m.num_col());
    for (size_t r = 0; r < result.rows; ++r)
        for (size_t c = 0; c < result.columns; ++c) result.at(r, c) = host.elem(r, c);
    return result;
}

#ifdef CUDA

void curand_ok(curandStatus_t result, const char* operation) {
    require(result == CURAND_STATUS_SUCCESS,
            std::string(operation) + " status=" + std::to_string(int(result)));
}

struct ReferenceGenerator {
    curandGenerator_t generator = nullptr;
    float* device = nullptr;
    ~ReferenceGenerator() {
        if (device) cudaFree(device);
        if (generator) curandDestroyGenerator(generator);
    }
};

Grid native_reference(int seed, bool normal, size_t rows, size_t columns) {
    ReferenceGenerator reference;
    curand_ok(curandCreateGenerator(&reference.generator, CURAND_RNG_PSEUDO_DEFAULT),
              "reference create");
    curand_ok(curandSetPseudoRandomGeneratorSeed(reference.generator, seed), "reference seed");
    curand_ok(curandSetGeneratorOffset(reference.generator, 0), "reference offset");
    const size_t count = rows * columns;
    // Juzhen pads odd requests to an even size (required for normal generation).
    const size_t generated_count = count + count % 2;
    CudaErrorCheck(cudaMalloc(reinterpret_cast<void**>(&reference.device),
                              generated_count * sizeof(float)));
    if (normal)
        curand_ok(curandGenerateNormal(reference.generator, reference.device, generated_count, 0.f, 1.f),
                  "reference normal");
    else
        curand_ok(curandGenerateUniform(reference.generator, reference.device, generated_count),
                  "reference uniform");
    std::vector<float> output(count);
    CudaErrorCheck(cudaMemcpy(output.data(), reference.device, count * sizeof(float),
                              cudaMemcpyDeviceToHost));
    Grid result(rows, columns);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < columns; ++c)
            result.at(r, c) = output[c * rows + r];
    return result;
}

void seed_case(int seed, bool normal, size_t rows, size_t columns) {
    const auto expected = native_reference(seed, normal, rows, columns);
    const std::string tag = "seed=" + std::to_string(seed) + (normal ? " normal " : " uniform ") +
                            std::to_string(rows) + "x" + std::to_string(columns);
    {
        GPUSampler sampler(seed);
        exact(draw(normal, rows, columns), expected, tag + " constructor matches cuRAND");
        consume_both_distributions();
        // An old explicit offset must not survive a reseed.
        curand_ok(curandSetGeneratorOffset(GPUSampler::gen, 64), "perturb offset");
        sampler.setseed(seed);
        exact(draw(normal, rows, columns), expected, tag + " reseed matches fresh constructor");
        consume_both_distributions();
        sampler.setseed(seed);
        exact(draw(normal, rows, columns), expected, tag + " repeated reseed repeats");
    }
    // GPUSampler owns a static generator: keep these lifetimes disjoint.
    {
        GPUSampler sampler(seed + 101);
        consume_both_distributions();
        sampler.setseed(seed);
        exact(draw(normal, rows, columns), expected, tag + " independent of initial seed");
    }
}

void seed_checks() {
    for (int seed : {0, 7, 733})
        for (bool normal : {false, true}) {
            seed_case(seed, normal, 2, 8);
            seed_case(seed, normal, 3, 5);
            seed_case(seed, normal, 1, 1);
        }
    std::cout << "[PASS] CUDA reseeding agrees with independent cuRAND references\n";
}
#else
// The first draw after a fresh sampler is the reference: reseeding must
// reproduce it whatever was drawn or seeded before, for both distributions.
void seed_case(int seed, bool normal, size_t rows, size_t columns) {
    const std::string tag = "seed=" + std::to_string(seed) + (normal ? " normal " : " uniform ") +
                            std::to_string(rows) + "x" + std::to_string(columns);
    Grid expected(0, 0);
    {
        GPUSampler sampler(seed);
        expected = host_grid(draw(normal, rows, columns));
        for (float v : expected.values)
            require(std::isfinite(v) && (normal || (v >= 0.f && v < 1.f)), tag + " value range");
        consume_both_distributions();
        sampler.setseed(seed);
        exact(draw(normal, rows, columns), expected, tag + " reseed matches fresh constructor");
        consume_both_distributions();
        sampler.setseed(seed);
        exact(draw(normal, rows, columns), expected, tag + " repeated reseed repeats");
    }
    {
        GPUSampler sampler(seed + 101);
        const auto other = host_grid(draw(normal, rows, columns));
        require(rows * columns < 4 || other.values != expected.values, tag + " seeds differ");
        consume_both_distributions();
        sampler.setseed(seed);
        exact(draw(normal, rows, columns), expected, tag + " independent of initial seed");
    }
}

void seed_checks() {
    for (int seed : {0, 7, 733})
        for (bool normal : {false, true}) {
            seed_case(seed, normal, 2, 8);
            seed_case(seed, normal, 3, 5);
            seed_case(seed, normal, 1, 1);
        }
    std::cout << "[PASS] Metal reseeding restarts both distributions\n";
}
#endif
} // namespace

int compute() {
    try {
        const char* requested = std::getenv("JUZHEN_CUDA_CONTRACT_CASE");
        const std::string selected = requested ? requested : "all";
        require(selected == "all" || selected == "seed" || selected == "slice",
                "JUZHEN_CUDA_CONTRACT_CASE must be all, seed, or slice");
        if (selected == "all" || selected == "seed") seed_checks();
        if (selected == "all" || selected == "slice") slice_checks();
        device_sync();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
#endif
