#!/usr/bin/env python3
"""Run the Poisson/Bernoulli GAM experiments in a new output folder."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'results/reproduced/figures58')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without writing or executing')
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser()
    if output.exists() or output.is_symlink():
        parser.error('Output already exists; choose a new directory.')
    output = output.resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'results/reproduced'):
        parser.error('Inside this repository, outputs must be under results/reproduced/.')
    jobs = []
    for family in ('poisson', 'bernoulli'):
        source = ROOT / 'experiments' / ('gam_' + family + '.py')
        if not source.is_file():
            parser.error('Missing experiment script: ' + str(source))
        folder = output / family
        command = [sys.executable, '-B', '-u', str(folder / source.name)]
        jobs.append((source, folder, command))
        print('Working directory:', folder)
        print(shlex.join(command))
    if args.dry_run:
        print('Dry run only; children will use Agg and one computational thread.')
        return 0

    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(output / 'matplotlib_cache'))
    env.update({key: '1' for key in THREADS})
    for source, folder, command in jobs:
        folder.mkdir(exist_ok=False)
        shutil.copyfile(source, folder / source.name)
        log = folder / 'python.log'
        print('Running', source.name, '— log:', log, flush=True)
        with log.open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=folder, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            parser.exit(1, 'Execution failed; retained outputs and log: ' + str(log) + '\n')
    print('Completed Figures 5–8. PDFs are in each family\'s gam_results_* directory.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
