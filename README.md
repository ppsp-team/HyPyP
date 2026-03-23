# HyPyP 🐍〰️🐍

The **Hy**perscanning **Py**thon **P**ipeline

[![PyPI version shields.io](https://img.shields.io/pypi/v/hypyp.svg)](https://pypi.org/project/HyPyP/) [![CI](https://github.com/ppsp-team/HyPyP/actions/workflows/Build.yml/badge.svg)](https://github.com/ppsp-team/HyPyP/actions/workflows/Build.yml) <a href="https://hypyp.readthedocs.io"><img src="https://readthedocs.org/projects/hypyp/badge/?version=latest"></a> [![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Mattermost](https://img.shields.io/discord/1065810348944408616?color=blue)](https://discord.gg/zYzjeGj7D6)

⚠️ This software is in beta and thus should be considered with caution. While we have done our best to test all the functionalities, there is no guarantee that the pipeline is entirely bug-free.

📖 See our [paper](https://academic.oup.com/scan/advance-article/doi/10.1093/scan/nsaa141/5919711) for more explanation and our plan for upcoming functionalities (aka Roadmap).

🤝 If you want to help you can submit bugs and suggestions of enhancements in our Github [Issues section](https://github.com/ppsp-team/HyPyP/issues).

🤓 For the motivated contributors, you can even help directly in the development of HyPyP. You will need to install [uv](https://docs.astral.sh/uv/) (see section below).

## Contributors

Original authors: Florence BRUN, Anaël AYROLLES, Phoebe CHEN, Amir DJALOVSKI, Yann BEAUXIS, Suzanne DIKKER, Guillaume DUMAS
New contributors: Ryssa MOFFAT, Marine Gautier MARTINS, Rémy RAMADOUR, Patrice FORTIN, Ghazaleh RANJBARAN, Quentin MOREAU, Caitriona DOUGLAS, Franck PORTEOUS, Jonas MAGO, Juan C. AVENDANO, Julie BONNAIRE, Martín A. MIGUEL, [@m2march](https://github.com/m2march) (ACCorr GPU/numba optimizations, BrainHack Montréal 2026)

## Installation

`pip install HyPyP`

## Documentation

HyPyP documentation of all the API functions is available online at [hypyp.readthedocs.io](https://hypyp.readthedocs.io/)

For getting started with HyPyP, we have designed a little walkthrough: [getting_started.ipynb](https://github.com/ppsp-team/HyPyP/blob/master/tutorial/getting_started.ipynb)

## Core API

🛠 [io.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/io.py) — Loaders (Florence, Anaël, Ghazaleh, Franck, Jonas, Guillaume)

🧰 [utils.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/utils.py) — Basic tools (Amir, Florence, Guillaume)

⚙️ [prep.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/prep.py) — Preprocessing (ICA & AutoReject) (Anaël, Florence, Guillaume)

🔠 [analyses.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/analyses.py) — Power spectral density and wide choice of connectivity measures (Phoebe, Suzanne, Florence, Ghazaleh, Juan, Guillaume)

📈 [stats.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/stats.py) — Statistics (permutations & cluster statistics) (Florence, Guillaume)

🧠 [viz.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/viz.py) — Inter-brain visualization (Anaël, Amir, Florence, Guillaume)

🎓 [Tutorials](https://github.com/ppsp-team/HyPyP/tree/master/tutorial) - Examples & documentation (Anaël, Florence, Yann, Ghazaleh, Caitriona, Guillaume)

## fNIRS hyperscanning

🔦 [fnirs/\*.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/fnirs) — Functional Near Infrared Spectroscopy hyperscanning features (Patrice)

🌊 [wavelet/\*.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/wavelet) — Continuous Wavelet Transform and Wavelet Transform Coherence (Patrice)

📊 [shiny/\*.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/app) — Shiny dashboards, install using `uv sync --extra shiny` (Patrice)

## Developer Installation (uv)

To develop HyPyP, we recommend using [uv](https://docs.astral.sh/uv/). Follow these steps:

### 1. Install uv:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or with pip
pip install uv
```

### 2. Clone the Repository:

```bash
git clone git@github.com:ppsp-team/HyPyP.git
cd HyPyP
```

### 3. Install Dependencies:

```bash
uv sync --group dev
```

### 4. Launch Jupyter Lab to Run Notebooks:

```bash
uv run jupyter lab
```

### Optional extras:

```bash
# Shiny dashboards
uv sync --group dev --extra shiny

# Numba optimization backend (CPU JIT)
uv sync --group dev --extra numba

# PyTorch optimization backend (GPU/MPS)
uv sync --group dev --extra torch
```

### VS Code Integration

uv creates a `.venv` directory in the project root by default. VS Code should auto-detect it. If not, point the Python interpreter to `.venv/bin/python`.

## Child Head Visualization

As of version 0.5.0b5, hypyp now supports visualization of parent-child or adult-child hyperscanning data. This allows for properly scaled and positioned head models when analyzing data from participants of different ages.

To use this functionality, simply set the `children=True` parameter in visualization functions and specify which participant is the child using the `child_head` parameter.

Example:

```python
# Visualize parent-child data (epo1 = parent, epo2 = child)
viz_3D_inter(epo1, epo2, C, threshold=0.95, steps=10, children=True, child_head=True)
```

# License

This project is licensed under the BSD 3-Clause License. See the license for details.
