# vi-glm-estimation

Code for **Beyond Maximum Likelihood: Variational Inequality Estimation for
Generalized Linear Models**, Linglingzhi Zhu, Jonghyeok Lee and Yao Xie,
NeurIPS 2026.

Experiments for Tables 2–6 and Figures 2–10.

## Run an experiment

- **MATLAB R2024a + Statistics and Machine Learning Toolbox:** add `scripts/`
  to the MATLAB path, then call `run_table2`,
  `run_table3`, `run_table4`, `run_table5`, or `run_figures234`.
- **Python 3.9.6:** install [requirements.txt](requirements.txt), then run
  `python scripts/run_figures58.py` or `python scripts/run_table6.py`.

See [EXPERIMENTS.md](EXPERIMENTS.md) for setup, commands and settings.
Outputs are saved under `results/reproduced/`. Use a new output folder for each run.

## Citation and license

[CITATION.cff](CITATION.cff) · [MIT License](LICENSE)

Bike Sharing data: Fanaee-T, H. (2013), UCI Machine Learning Repository,
[doi:10.24432/C5W894](https://doi.org/10.24432/C5W894),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Dataset details](experiments/data/bike_sharing/Readme.txt).
