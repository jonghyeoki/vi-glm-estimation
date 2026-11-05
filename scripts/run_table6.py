#!/usr/bin/env python3
"""Run the converged softplus experiments with the Table 6 settings."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'results/reproduced/table6')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without writing or executing')
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser()
    if output.exists() or output.is_symlink():
        parser.error('Output already exists; choose a new directory.')
    output = output.resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'results/reproduced'):
        parser.error('Inside this repository, outputs must be under results/reproduced/.')
    scripts = ROOT / 'experiments'
    data = ROOT / 'experiments/data/bike_sharing/hour.csv'
    synthetic = scripts / 'converged_softplus_comparison.py'
    bike = scripts / 'bike_softplus_converged.py'
    if not all(p.is_file() for p in (synthetic, bike, data)):
        parser.error('An experiment script or the included hour.csv is missing.')
    jobs = [
        ('synthetic', [sys.executable, '-B', '-u', str(synthetic),
            '--dimension', '20', '--sample-sizes', '400', '--budgets', '20', '50', '100', '200',
            '--repetitions', '1000', '--seed', '7', '--step0', '0.01', '--decay', '0.975',
            '--vi-tolerance', '1e-8', '--mle-tolerance', '1e-8', '--max-iterations', '5000',
            '--parameter-type', 'dense_unit', '--output-dir', str(output / 'synthetic')]),
        ('bike', [sys.executable, '-B', '-u', str(bike), '--data', str(data),
                  '--output-dir', str(output / 'bike')]),
    ]
    for name, command in jobs:
        print('Working directory:', output / name)
        print(shlex.join(command))
    if args.dry_run:
        print('Dry run only; children will use Agg and one computational thread.')
        return 0

    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(output / 'matplotlib_cache'))
    env.update({key: '1' for key in THREADS})
    for name, command in jobs:
        folder = output / name
        folder.mkdir(exist_ok=False)
        log = folder / 'python.log'
        print('Running', name, '— log:', log, flush=True)
        with log.open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=folder, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            parser.exit(1, 'Execution failed; retained outputs and log: ' + str(log) + '\n')
    print('Completed Table 6 experiments.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
