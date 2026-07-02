/**
 * @file demo_simulation.cu
 * @brief Simple 3D N-body galaxy-formation toy simulation with periodic PPM frames.
 *
 * The simulation starts from a rotating stellar disk plus a compact bulge.
 * Gravity, softening, and a leapfrog-style update drive the evolution in 3D.
 * Every few steps the current density field is projected to 2D and rasterized
 * into a PPM image.
 */

#include "../cpp/juzhen.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace fs = std::filesystem;

#ifdef CUDA
#define FLOAT CUDAfloat
#elif defined(APPLE_SILICON)
#define FLOAT MPSfloat
#else
#define FLOAT float
#endif

template <class D>
static Matrix<float> as_host(const Matrix<D>& m) { return m.to_host(); }
template <>
Matrix<float> as_host<float>(const Matrix<float>& m) { return m; }

struct SimulationState {
    Matrix<FLOAT> x;
    Matrix<FLOAT> y;
    Matrix<FLOAT> z;
    Matrix<FLOAT> vx;
    Matrix<FLOAT> vy;
    Matrix<FLOAT> vz;
    Matrix<FLOAT> mass;
};

struct GridField {
    int nx;
    int ny;
    int nz;
    int cells;
    float view_radius;
    float cell_size;
    Matrix<FLOAT> cx;
    Matrix<FLOAT> cy;
    Matrix<FLOAT> cz;
    Matrix<FLOAT> halo_mass;
};

struct PMWorkspace {
    Matrix<FLOAT> kx;
    Matrix<FLOAT> ky;
    Matrix<FLOAT> kz;
    Matrix<FLOAT> cell_mass;
    Matrix<FLOAT> total_mass;
    Matrix<FLOAT> ax_grid;
    Matrix<FLOAT> ay_grid;
    Matrix<FLOAT> az_grid;
    Matrix<FLOAT> ax_part;
    Matrix<FLOAT> ay_part;
    Matrix<FLOAT> az_part;
};

#ifdef CUDA
__global__ static void deposit_mass_kernel(float* cell_mass,
                                           const float* x,
                                           const float* y,
                                           const float* z,
                                           const float* mass,
                                           int n,
                                           int nx,
                                           int ny,
                                           int nz,
                                           float view_radius,
                                           float cell_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    float px = x[i];
    float py = y[i];
    float pz = z[i];
    int gx = (int)floorf((px + view_radius) / cell_size);
    int gy = (int)floorf((py + view_radius) / cell_size);
    int gz = (int)floorf((pz + view_radius) / cell_size);
    gx = max(0, min(nx - 1, gx));
    gy = max(0, min(ny - 1, gy));
    gz = max(0, min(nz - 1, gz));
    int idx = (gz * ny + gy) * nx + gx;
    atomicAdd(cell_mass + idx, mass[i]);
}

__global__ static void sample_accel_kernel(float* ax_out,
                                           float* ay_out,
                                           float* az_out,
                                           const float* x,
                                           const float* y,
                                           const float* z,
                                           int n,
                                           const float* ax_grid,
                                           const float* ay_grid,
                                           const float* az_grid,
                                           int nx,
                                           int ny,
                                           int nz,
                                           float view_radius,
                                           float cell_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    float fx = (x[i] + view_radius) / cell_size - 0.5f;
    float fy = (y[i] + view_radius) / cell_size - 0.5f;
    float fz = (z[i] + view_radius) / cell_size - 0.5f;

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int z0 = (int)floorf(fz);
    float tx = fx - x0;
    float ty = fy - y0;
    float tz = fz - z0;

    x0 = max(0, min(nx - 1, x0));
    y0 = max(0, min(ny - 1, y0));
    z0 = max(0, min(nz - 1, z0));
    int x1 = max(0, min(nx - 1, x0 + 1));
    int y1 = max(0, min(ny - 1, y0 + 1));
    int z1 = max(0, min(nz - 1, z0 + 1));

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int dz_i = 0; dz_i <= 1; ++dz_i) {
        int gz = dz_i ? z1 : z0;
        float wz = dz_i ? tz : (1.0f - tz);
        for (int dy_i = 0; dy_i <= 1; ++dy_i) {
            int gy = dy_i ? y1 : y0;
            float wy = dy_i ? ty : (1.0f - ty);
            for (int dx_i = 0; dx_i <= 1; ++dx_i) {
                int gx = dx_i ? x1 : x0;
                float wx = dx_i ? tx : (1.0f - tx);
                float w = wx * wy * wz;
                int idx = (gz * ny + gy) * nx + gx;
                ax += w * ax_grid[idx];
                ay += w * ay_grid[idx];
                az += w * az_grid[idx];
            }
        }
    }

    ax_out[i] = ax;
    ay_out[i] = ay;
    az_out[i] = az;
}
#endif

static float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

static string frame_path(const string& dir, int frame_id) {
    char buf[64];
    snprintf(buf, sizeof(buf), "frame_%04d.ppm", frame_id);
    return dir + "/" + buf;
}

static bool encode_videos(const string& output_dir) {
    const string mp4_path = output_dir + "/galaxy_sim.mp4";
    const string gif_path = output_dir + "/galaxy_sim.gif";

    std::ostringstream mp4_cmd;
    mp4_cmd
        << "ffmpeg -y -framerate 12 -i " << output_dir << "/frame_%04d.ppm "
        << "-c:v libx264 -pix_fmt yuv420p -movflags +faststart "
        << mp4_path << " >/dev/null 2>&1";

    std::ostringstream gif_cmd;
    gif_cmd
        << "ffmpeg -y -framerate 12 -i " << output_dir << "/frame_%04d.ppm "
        << "-vf \"fps=12,scale=768:-1:flags=lanczos,split[s0][s1];"
        << "[s0]palettegen[p];[s1][p]paletteuse\" "
        << gif_path << " >/dev/null 2>&1";

    int mp4_rc = std::system(mp4_cmd.str().c_str());
    if (mp4_rc != 0) {
        std::cerr << "Warning: failed to generate MP4 via ffmpeg." << std::endl;
        return false;
    }

    int gif_rc = std::system(gif_cmd.str().c_str());
    if (gif_rc != 0) {
        std::cerr << "Warning: failed to generate GIF via ffmpeg." << std::endl;
        return false;
    }

    std::cout << "Generated video: " << mp4_path << "\n";
    std::cout << "Generated gif:   " << gif_path << "\n";
    return true;
}

static void save_frame_ppm(const string& path,
                           const Matrix<float>& x,
                           const Matrix<float>& y,
                           const Matrix<float>& z,
                           const Matrix<float>& vx,
                           const Matrix<float>& vy,
                           const Matrix<float>& vz,
                           const Matrix<float>& mass,
                           int width,
                           int height,
                           float view_radius) {
    vector<float> density(width * height, 0.0f);
    vector<float> heat(width * height, 0.0f);

    const int n = (int)x.num_row();
    for (int i = 0; i < n; ++i) {
        const float px_world = x.elem(i, 0);
        const float py_world = y.elem(i, 0);
        const float pz_world = z.elem(i, 0);
        const float pvx = vx.elem(i, 0);
        const float pvy = vy.elem(i, 0);
        const float pvz = vz.elem(i, 0);
        const float pmass = mass.elem(i, 0);

        float px = (px_world / view_radius * 0.5f + 0.5f) * (width - 1);
        float py = (py_world / view_radius * 0.5f + 0.5f) * (height - 1);
        int ix = (int)std::lround(px);
        int iy = (int)std::lround(py);
        if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue;

        float speed = std::sqrt(pvx * pvx + pvy * pvy + pvz * pvz);
        float depth = clamp01(0.5f + 0.5f * pz_world / view_radius);
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int xx = ix + dx;
                int yy = iy + dy;
                if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;
                float r2 = float(dx * dx + dy * dy);
                float w = std::exp(-0.6f * r2) * pmass * (0.65f + 0.35f * depth);
                int idx = yy * width + xx;
                density[idx] += w;
                heat[idx] += w * speed;
            }
        }
    }

    float max_density = 1e-6f;
    float max_heat = 1e-6f;
    for (size_t i = 0; i < density.size(); ++i) {
        max_density = std::max(max_density, density[i]);
        max_heat = std::max(max_heat, heat[i]);
    }

    vector<unsigned char> image(width * height * 3, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float d = std::log1p(18.0f * density[idx] / max_density) / std::log1p(18.0f);
            float h = std::sqrt(heat[idx] / max_heat);

            float r = clamp01(0.10f + 1.25f * d + 0.55f * h);
            float g = clamp01(0.06f + 0.95f * d + 0.22f * h);
            float b = clamp01(0.10f + 0.85f * d - 0.12f * h);

            if (d > 0.02f) {
                float glow = clamp01((d - 0.02f) * 1.6f);
                b = clamp01(b + 0.18f * glow);
            }

            int out = idx * 3;
            image[out + 0] = (unsigned char)std::lround(255.0f * r);
            image[out + 1] = (unsigned char)std::lround(255.0f * g);
            image[out + 2] = (unsigned char)std::lround(255.0f * b);
        }
    }

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        std::cerr << "Cannot write frame: " << path << std::endl;
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(image.data(), 1, image.size(), fp);
    fclose(fp);
}

static GridField build_grid_field(int nx,
                                  int ny,
                                  int nz,
                                  float view_radius,
                                  float halo_mass_scale,
                                  float halo_core_radius) {
    const int cells = nx * ny * nz;
    const float cell_size = 2.0f * view_radius / static_cast<float>(nx);
    auto cx_h = Matrix<float>::zeros(cells, 1);
    auto cy_h = Matrix<float>::zeros(cells, 1);
    auto cz_h = Matrix<float>::zeros(cells, 1);
    auto halo_mass_h = Matrix<float>::zeros(cells, 1);
    const float cell_volume = cell_size * cell_size * cell_size;

    for (int gz = 0; gz < nz; ++gz) {
        for (int gy = 0; gy < ny; ++gy) {
            for (int gx = 0; gx < nx; ++gx) {
                const int idx = (gz * ny + gy) * nx + gx;
                const float cx = -view_radius + (gx + 0.5f) * cell_size;
                const float cy = -view_radius + (gy + 0.5f) * cell_size;
                const float cz = -view_radius + (gz + 0.5f) * cell_size;
                cx_h.elem(idx, 0) = cx;
                cy_h.elem(idx, 0) = cy;
                cz_h.elem(idx, 0) = cz;

                // Simple cored halo density profile in 3D.
                const float r2 = cx * cx + cy * cy + cz * cz;
                const float rho = halo_mass_scale / (1.0f + r2 / (halo_core_radius * halo_core_radius));
                halo_mass_h.elem(idx, 0) = rho * cell_volume;
            }
        }
    }

    return {
        nx,
        ny,
        nz,
        cells,
        view_radius,
        cell_size,
        Matrix<FLOAT>(cx_h),
        Matrix<FLOAT>(cy_h),
        Matrix<FLOAT>(cz_h),
        Matrix<FLOAT>(halo_mass_h)
    };
}

static PMWorkspace build_pm_workspace(const GridField& grid, float softening) {
    auto cx_h = as_host(grid.cx);
    auto cy_h = as_host(grid.cy);
    auto cz_h = as_host(grid.cz);
    auto kx_h = Matrix<float>::zeros(grid.cells, grid.cells);
    auto ky_h = Matrix<float>::zeros(grid.cells, grid.cells);
    auto kz_h = Matrix<float>::zeros(grid.cells, grid.cells);

    const float mesh_soft = std::max(softening, 0.5f * grid.cell_size);
    for (int i = 0; i < grid.cells; ++i) {
        const float tx = cx_h.elem(i, 0);
        const float ty = cy_h.elem(i, 0);
        const float tz = cz_h.elem(i, 0);
        for (int j = 0; j < grid.cells; ++j) {
            if (i == j) continue;
            const float dx = cx_h.elem(j, 0) - tx;
            const float dy = cy_h.elem(j, 0) - ty;
            const float dz = cz_h.elem(j, 0) - tz;
            const float r2 = dx * dx + dy * dy + dz * dz + mesh_soft * mesh_soft;
            const float inv_r = 1.0f / std::sqrt(r2);
            const float inv_r3 = inv_r * inv_r * inv_r;
            kx_h.elem(i, j) = dx * inv_r3;
            ky_h.elem(i, j) = dy * inv_r3;
            kz_h.elem(i, j) = dz * inv_r3;
        }
    }

    return {
        Matrix<FLOAT>(kx_h),
        Matrix<FLOAT>(ky_h),
        Matrix<FLOAT>(kz_h),
        Matrix<FLOAT>::zeros(grid.cells, 1),
        Matrix<FLOAT>::zeros(grid.cells, 1),
        Matrix<FLOAT>::zeros(grid.cells, 1),
        Matrix<FLOAT>::zeros(grid.cells, 1),
        Matrix<FLOAT>::zeros(grid.cells, 1),
        Matrix<FLOAT>::zeros(0, 0),
        Matrix<FLOAT>::zeros(0, 0),
        Matrix<FLOAT>::zeros(0, 0)
    };
}

static SimulationState initialize_particles(int count, float disk_radius) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> unit01(0.0f, 1.0f);
    std::uniform_real_distribution<float> angle01(0.0f, 2.0f * 3.1415926535f);
    std::normal_distribution<float> bulge_noise(0.0f, 0.12f * disk_radius);
    std::normal_distribution<float> disk_jitter(0.0f, 0.03f);
    std::normal_distribution<float> disk_thickness(0.0f, 0.10f * disk_radius);

    const int bulge_count = count / 8;
    const float disk_mass = 1.0f;
    const float bulge_mass = 3.5f;
    const float central_soft_mass = 14.0f;

    auto x_h = Matrix<float>::zeros(count, 1);
    auto y_h = Matrix<float>::zeros(count, 1);
    auto z_h = Matrix<float>::zeros(count, 1);
    auto vx_h = Matrix<float>::zeros(count, 1);
    auto vy_h = Matrix<float>::zeros(count, 1);
    auto vz_h = Matrix<float>::zeros(count, 1);
    auto mass_h = Matrix<float>::zeros(count, 1);

    for (int i = 0; i < count; ++i) {
        if (i < bulge_count) {
            x_h.elem(i, 0) = bulge_noise(rng);
            y_h.elem(i, 0) = bulge_noise(rng);
            z_h.elem(i, 0) = bulge_noise(rng);
            vx_h.elem(i, 0) = 0.0f;
            vy_h.elem(i, 0) = 0.0f;
            vz_h.elem(i, 0) = 0.0f;
            mass_h.elem(i, 0) = bulge_mass;
        } else {
            float u = unit01(rng);
            float radius = disk_radius * std::sqrt(u);
            float theta = angle01(rng);

            // Mild three-arm perturbation to help visible structure form.
            radius *= 0.92f + 0.08f * std::cos(3.0f * theta);
            x_h.elem(i, 0) = radius * std::cos(theta);
            y_h.elem(i, 0) = radius * std::sin(theta);
            z_h.elem(i, 0) = disk_thickness(rng);

            float enclosed_mass = central_soft_mass + disk_mass * (radius / disk_radius) * count;
            float speed = std::sqrt(std::max(0.0f, 0.18f * enclosed_mass / (radius + 0.15f)));
            vx_h.elem(i, 0) = -std::sin(theta) * speed + disk_jitter(rng);
            vy_h.elem(i, 0) =  std::cos(theta) * speed + disk_jitter(rng);
            vz_h.elem(i, 0) = 0.10f * disk_jitter(rng);
            mass_h.elem(i, 0) = disk_mass;
        }
    }

    return {
        Matrix<FLOAT>(x_h),
        Matrix<FLOAT>(y_h),
        Matrix<FLOAT>(z_h),
        Matrix<FLOAT>(vx_h),
        Matrix<FLOAT>(vy_h),
        Matrix<FLOAT>(vz_h),
        Matrix<FLOAT>(mass_h)
    };
}

static void update_particles(SimulationState& state,
                             const GridField& grid,
                             PMWorkspace& work,
                             float dt,
                             float softening,
                             float gravity,
                             float central_mass) {
    const int n = (int)state.x.num_row();
    if ((int)work.ax_part.num_row() != n) {
        work.ax_part = Matrix<FLOAT>::zeros(n, 1);
        work.ay_part = Matrix<FLOAT>::zeros(n, 1);
        work.az_part = Matrix<FLOAT>::zeros(n, 1);
    }

    {
        const int cells = grid.cells;
        auto x_h = as_host(state.x);
        auto y_h = as_host(state.y);
        auto z_h = as_host(state.z);
        auto mass_h = as_host(state.mass);
        auto cell_mass_h = Matrix<float>::zeros(cells, 1);

        for (int i = 0; i < n; ++i) {
            float px = x_h.elem(i, 0);
            float py = y_h.elem(i, 0);
            float pz = z_h.elem(i, 0);
            int gx = (int)std::floor((px + grid.view_radius) / grid.cell_size);
            int gy = (int)std::floor((py + grid.view_radius) / grid.cell_size);
            int gz = (int)std::floor((pz + grid.view_radius) / grid.cell_size);
            gx = std::max(0, std::min(grid.nx - 1, gx));
            gy = std::max(0, std::min(grid.ny - 1, gy));
            gz = std::max(0, std::min(grid.nz - 1, gz));
            int idx = (gz * grid.ny + gy) * grid.nx + gx;
            cell_mass_h.elem(idx, 0) += mass_h.elem(i, 0);
        }

        work.cell_mass = Matrix<FLOAT>(cell_mass_h);
        work.total_mass = work.cell_mass + grid.halo_mass;
        work.ax_grid = gravity * (work.kx * work.total_mass);
        work.ay_grid = gravity * (work.ky * work.total_mass);
        work.az_grid = gravity * (work.kz * work.total_mass);

        auto ax_grid_h = as_host(work.ax_grid);
        auto ay_grid_h = as_host(work.ay_grid);
        auto az_grid_h = as_host(work.az_grid);
        auto ax_h = Matrix<float>::zeros(n, 1);
        auto ay_h = Matrix<float>::zeros(n, 1);
        auto az_h = Matrix<float>::zeros(n, 1);
        for (int i = 0; i < n; ++i) {
            int gx = (int)std::floor((x_h.elem(i, 0) + grid.view_radius) / grid.cell_size);
            int gy = (int)std::floor((y_h.elem(i, 0) + grid.view_radius) / grid.cell_size);
            int gz = (int)std::floor((z_h.elem(i, 0) + grid.view_radius) / grid.cell_size);
            gx = std::max(0, std::min(grid.nx - 1, gx));
            gy = std::max(0, std::min(grid.ny - 1, gy));
            gz = std::max(0, std::min(grid.nz - 1, gz));
            int idx = (gz * grid.ny + gy) * grid.nx + gx;
            ax_h.elem(i, 0) = ax_grid_h.elem(idx, 0);
            ay_h.elem(i, 0) = ay_grid_h.elem(idx, 0);
            az_h.elem(i, 0) = az_grid_h.elem(idx, 0);
        }
        work.ax_part = Matrix<FLOAT>(ax_h);
        work.ay_part = Matrix<FLOAT>(ay_h);
        work.az_part = Matrix<FLOAT>(az_h);
    }

    auto r2c = square(state.x) + square(state.y) + square(state.z) + softening * softening;
    auto inv_rc = Matrix<FLOAT>::ones(n, 1) / sqrt(Matrix<FLOAT>(r2c));
    auto inv_rc3 = hadmd(inv_rc, hadmd(Matrix<FLOAT>(inv_rc), Matrix<FLOAT>(inv_rc)));
    auto ax_central = -gravity * central_mass * hadmd(state.x, inv_rc3);
    auto ay_central = -gravity * central_mass * hadmd(state.y, inv_rc3);
    auto az_central = -gravity * central_mass * hadmd(state.z, inv_rc3);

    auto ax = work.ax_part + ax_central;
    auto ay = work.ay_part + ay_central;
    auto az = work.az_part + az_central;

    state.vx += dt * ax;
    state.vy += dt * ay;
    state.vz += dt * az;
    state.x += dt * state.vx;
    state.y += dt * state.vy;
    state.z += dt * state.vz;
}

static float kinetic_energy(const SimulationState& state) {
    auto speed2 = hadmd(state.vx, state.vx) + hadmd(state.vy, state.vy) + hadmd(state.vz, state.vz);
    auto kinetic = 0.5f * hadmd(state.mass, speed2);
    return as_host(sum(kinetic, 0)).elem(0, 0);
}

int compute() {
    global_rand_gen.seed(42);
#ifdef CUDA
    GPUSampler sampler(42);
#endif

    const int particle_count = 1024;
    const int steps = 2400;
    const int snapshot_every = 24;
    const int width = 768;
    const int height = 768;
    const float disk_radius = 5.5f;
    const float view_radius = 7.5f;
    const int grid_n = 16;
    const float dt = 0.008f;
    const float softening = 0.09f;
    const float gravity = 0.18f;
    const float central_mass = 48.0f;
    const float halo_mass_scale = 11.0f;
    const float halo_core_radius = 4.2f;
    const string output_dir = "res/galaxy_sim";

    fs::create_directories(output_dir);
    auto state = initialize_particles(particle_count, disk_radius);
    auto grid = build_grid_field(grid_n, grid_n, grid_n, view_radius, halo_mass_scale, halo_core_radius);
    auto pm = build_pm_workspace(grid, softening);

    cout << "=== N-body Galaxy Formation Demo ===\n";
    cout << "Particles: " << particle_count << "\n";
    cout << "Grid: " << grid_n << "x" << grid_n << "x" << grid_n << " cells\n";
    cout << "Steps: " << steps << ", snapshot every " << snapshot_every << " steps\n";
    cout << "dt=" << dt << ", softening=" << softening << ", gravity=" << gravity
         << ", central_mass=" << central_mass << "\n";
    cout << "Dark-matter halo: scale=" << halo_mass_scale
         << ", core_radius=" << halo_core_radius << "\n";
    cout << "Output directory: " << output_dir << "\n\n";

    int frame_id = 0;
    save_frame_ppm(frame_path(output_dir, frame_id++),
                   as_host(state.x), as_host(state.y), as_host(state.z),
                   as_host(state.vx), as_host(state.vy), as_host(state.vz), as_host(state.mass),
                   width, height, view_radius);

    for (int step = 1; step <= steps; ++step) {
        update_particles(state, grid, pm, dt, softening, gravity, central_mass);

        if (step % snapshot_every == 0 || step == steps) {
            string path = frame_path(output_dir, frame_id++);
            auto x_h = as_host(state.x);
            auto y_h = as_host(state.y);
            auto z_h = as_host(state.z);
            auto vx_h = as_host(state.vx);
            auto vy_h = as_host(state.vy);
            auto vz_h = as_host(state.vz);
            auto m_h = as_host(state.mass);
            save_frame_ppm(path, x_h, y_h, z_h, vx_h, vy_h, vz_h, m_h, width, height, view_radius);

            float kinetic = kinetic_energy(state);
            cout << "step " << step << "/" << steps
                 << "  frame=" << path
                 << "  kinetic=" << kinetic << "\n";
        }
    }

    cout << "\nFinished. Frames written to " << output_dir << "\n";
    encode_videos(output_dir);
    return 0;
}