#include "../ml/dataloader.hpp"
#include <filesystem>

namespace {

// Deterministic, verifiable synthetic sample values.
// Feature value for row i of global sample j.
inline float synth_x(size_t i, size_t j) { return (float)((i + j) % 256); }
// Integer class label for global sample j.
inline int synth_y(size_t j) { return (int)(j % 10); }

// Write a small synthetic dataset in the DataLoader's .matrix format into a
// temp folder: train_x.matrix is d x n (column j = sample j), train_y.matrix
// is 1 x n integer labels. Returns the folder path.
std::string make_synthetic_dataset(size_t d, size_t n) {
    using namespace Juzhen;
    namespace fs = std::filesystem;

    fs::path dir = fs::temp_directory_path() / "juzhen_dataloader_synth";
    fs::create_directories(dir);

    Matrix<float> x("x", d, n);
    for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < d; i++)
            x(i, j) = synth_x(i, j);

    Matrix<int> y("y", 1, n);
    for (size_t j = 0; j < n; j++)
        y(0, j) = synth_y(j);

    write((dir / "train_x.matrix").string(), x);
    write((dir / "train_y.matrix").string(), y);
    return dir.string();
}

}  // namespace

int test1()
{
    using namespace Juzhen;

    const size_t d = 784;          // MNIST-shaped feature dimension
    const size_t n_total = 1000;   // small synthetic set (finishes in a few ms)
    const size_t batch_size = 34;  // leaves a partial final batch (1000 % 34 = 14)

    std::string folder = make_synthetic_dataset(d, n_total);

    const size_t rows_to_check[] = {0, d / 2, d - 1};
    size_t seen = 0;
    int ret = 7;  // set below; 7 == never reached a partial batch (unexpected)

    // Scope the loader so its destructor closes the .matrix files before we
    // delete them below (Windows won't remove files that are still open).
    {
        DataLoader<float, int> loader(folder, "train", batch_size);

        // Iterate one full epoch. Expect ceil(n_total / batch_size) batches,
        // the last of which is a partial batch.
        for (size_t b = 0; b < n_total / batch_size + 2; b++) {
            auto [x, y] = loader.next_batch();

            if (x.num_row() != d)           { ret = 1; break; }  // wrong feature dim
            if (x.num_col() != y.num_col()) { ret = 2; break; }  // x/y batch mismatch
            if (x.num_col() > batch_size)   { ret = 3; break; }  // oversized batch

            // Samples must come back in order, with exactly the values we wrote.
            bool content_ok = true;
            for (size_t c = 0; c < x.num_col() && content_ok; c++) {
                size_t global = seen + c;
                for (size_t i : rows_to_check)
                    if (x(i, c) != synth_x(i, global)) { ret = 4; content_ok = false; break; }
                if (content_ok && y(0, c) != synth_y(global)) { ret = 5; content_ok = false; }
            }
            if (!content_ok) break;

            seen += x.num_col();

            if (x.num_col() < batch_size) {
                // Final partial batch reached: total must equal the dataset size.
                ret = (seen == n_total) ? 0 : 6;
                break;
            }
        }
    }

    std::error_code ec;
    std::filesystem::remove_all(folder, ec);  // best-effort, non-throwing cleanup
    return ret;
}


int compute()
{
    spdlog::set_level(spdlog::level::debug);
    std::cout << __cplusplus << " " << HAS_CONCEPTS << std::endl;

    int ret = test1();
    std::cout << std::endl;

    if (ret == 0)
    {
        LOG_INFO("--------------------");
        LOG_INFO("|      ALL OK!     |");
        LOG_INFO("--------------------");
    }
    else
    {
        LOG_ERROR("--------------------");
        LOG_ERROR("|    NOT ALL OK!   |");
        LOG_ERROR("--------------------");
    }

    return ret;
}
