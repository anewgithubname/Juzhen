# Training checkpoints and AMD resume

Include `ml/checkpoint.hpp` to save and restore a network with
`save_checkpoint` and `load_checkpoint`. ROCm networks use the same interface
as CPU networks. AMD tensor data is copied between device memory and the file.
This is training-state restoration; saving weights alone does not restore Adam.

## Usage

Create the same network, in the same layer order and with the same architecture
settings, before loading. The checkpoint does not construct the model for you.
Call save after a complete optimizer step, and load before the next forward pass.
Use one training thread/writer per checkpoint path.

```cpp
#include "cpp/juzhen.hpp"
#include "ml/checkpoint.hpp"

using namespace Juzhen;
using D = ROCMfloat; // In a ROCm build; use float for CPU.

TransformerLayer<D> block(8, 8, 16, 4, 2, 2, true);
std::list<Layer<D>*> network{&block};
TrainingProgress progress{};
std::string error;

// On a later run, after constructing the same network:
if (resume_requested) {
    if (!load_checkpoint(network, "checkpoints/latest.bin", progress, &error))
        throw std::runtime_error(error);
    // Restore the data iterator to progress.data_position and rebuild any
    // application scheduler state before continuing the training loop.
}

// Inside the training loop, after a completed optimizer update:
++progress.step;
progress.epoch = current_epoch;
progress.data_position = next_sample_position;
std::filesystem::create_directories("checkpoints");
if (!save_checkpoint(network, "checkpoints/latest.bin", progress, &error))
    throw std::runtime_error(error);
```

`resume_requested`, `current_epoch` and `next_sample_position` belong to the
application. The progress fields are caller-managed counters; `step` should count
completed updates and `data_position` should identify the next data item.

## What is restored

- Parameters and optimizer states exposed by each layer's
  `checkpoint_parameters()` and `checkpoint_optimizers()` interfaces.
  Transformer exposes all 13 parameters and all 13 Adam states.
- Adam first/second moments, iteration counter, learning rate, beta values and epsilon.
- `epoch`, `step`, `data_position`, and the CPU `global_rand_gen` engine.

The existing version-2 format is unchanged. Load checks layer types, parameter
names, tensor dimensions, optimizer names and the completion marker. It stages
all state before updating the live network, so a rejected file preserves the
model, optimizer, progress and host RNG. Loading temporarily needs memory for
another copy of the saved model and optimizer state.

Save writes a temporary file in the same directory and replaces the destination
only after a successful write and close. A failed replacement leaves the old
destination intact. The temporary suffix is `.tmp`; concurrent writers using the
same path are unsupported. These checks do not detect every possible bit flip
inside tensor data; the format has no content checksum.

## Reproducibility limits

- **Device RNG state is not serialized.** Direct `Matrix<ROCMfloat>::rand()` /
  `randn()` calls use rocRAND, which keeps a separate state. CUDA's device RNG
  is also outside this format. Training can continue after loading, but future
  device-generated random values need not match uninterrupted training.
- For the deterministic resume path tested here, generate random inputs using
  `Matrix<float>::randn()` / `rand()` and then transfer them to the GPU. These
  draws use the saved host generator. This does not cover an external library's
  RNG or an independently created random engine.
- Dataset order, shuffling permutations, data-loader buffers, learning-rate
  schedules, gradient accumulation, model configuration and external state are
  application responsibilities. Only the three progress counters are stored;
  merely restoring a counter does not reposition an iterator.
- Save at an optimizer-step boundary. Activations and cached forward/JVP state
  are not saved; run a new forward pass after loading.
- Use the same backend, architecture settings, compiler ABI and compatible
  runtime to resume. Layer identifiers use C++ `typeid` names; this format is
  not a portable CPU-to-AMD model exchange format. Shape checks cannot catch
  configuration changes such as a different head count with identical shapes.

## Tests

```bash
cmake --build build_rocm --target testCheckpointResume
ctest --test-dir build_rocm --output-on-failure -R '^checkpoint_'
```

Use `build_cpu` for CPU. Test artifacts live inside each build's
`checkpoint_resume_tests` directory; the two builds do not share output files.
The restart tests use a CTest fixture, so selecting `checkpoint_restart_resume`
also schedules its preparation step.

- `checkpoint_resume_consistency`: linear and Transformer split training versus
  uninterrupted training, comparing all exposed parameters and Adam state.
  It also covers replacement, truncated/corrupt headers or footer, missing files,
  incompatible networks and unchanged live state after rejection.
- `checkpoint_restart_prepare`: saves a Transformer after two steps, then saves
  the expected state after three more uninterrupted steps.
- `checkpoint_restart_resume`: a new process loads the two-step checkpoint and
  performs those three steps. It compares all 13 parameters and Adam states,
  progress and the host RNG with the expected result.

These tests use actual ROCm tensors in a ROCm build and deliberately use host
random inputs. They do not claim to test restoration of the rocRAND stream.

Validated on 2026-09-07 with CPU/OpenBLAS and Radeon 8060S (`gfx1151`, ROCm 7.1):
all three tests passed on each backend. Linear, Transformer and separate-process
continuations each had maximum absolute parameter/Adam error 0. NVIDIA hardware
was not tested in this validation.
