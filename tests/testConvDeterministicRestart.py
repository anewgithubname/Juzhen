"""Run convolution checkpoint prepare/resume in independent processes."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--storage-mount', type=Path)
    args = p.parse_args()
    if args.storage_mount and not os.path.ismount(args.storage_mount):
        raise RuntimeError('Required storage drive is not mounted')
    output = args.output.resolve() / ('run-' + str(os.getpid()))
    output.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CONV_TEST_')}
    env.update(CONV_TEST_OUTPUT=str(output), NVIDIA_TF32_OVERRIDE='0', NVIDIA_TF32='0',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for phase in ('prepare', 'resume'):
        env['CONV_TEST_PHASE'] = phase
        result = subprocess.run([str(args.binary.resolve())], env=env)
        if result.returncode == 77:
            raise SystemExit(77)
        result.check_returncode()
    assert (output / 'continuous.ckpt').read_bytes() == (output / 'resumed.ckpt').read_bytes()
    print('PASS bitwise-identical convolution parameters, optimizer and RNG after process restart')


if __name__ == '__main__':
    main()
