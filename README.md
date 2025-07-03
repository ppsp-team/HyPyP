# HyPyP 🐍〰️🐍

The **Hy**perscanning **Py**thon **P**ipeline

[![PyPI version shields.io](https://img.shields.io/pypi/v/hypyp.svg)](https://pypi.org/project/HyPyP/) [![CI](https://github.com/ppsp-team/HyPyP/actions/workflows/Build.yml/badge.svg)](https://github.com/ppsp-team/HyPyP/actions/workflows/Build.yml) <a href="https://hypyp.readthedocs.io"><img src="https://readthedocs.org/projects/hypyp/badge/?version=latest"></a> [![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Mattermost](https://img.shields.io/discord/1065810348944408616?color=blue)](https://discord.gg/zYzjeGj7D6)

⚠️ This software is in beta and thus should be considered with caution. While we have done our best to test all the functionalities, there is no guarantee that the pipeline is entirely bug-free.

📖 See our [paper](https://academic.oup.com/scan/advance-article/doi/10.1093/scan/nsaa141/5919711) for more explanation and our plan for upcoming functionalities (aka Roadmap).

🤝 If you want to help you can submit bugs and suggestions of enhancements in our Github [Issues section](https://github.com/ppsp-team/HyPyP/issues).

🤓 For the motivated contributors, you can even help directly in the development of HyPyP. You will need to install [Poetry](https://python-poetry.org/) (see section below).

## Contributors

Original authors: Florence BRUN, Anaël AYROLLES, Phoebe CHEN, Amir DJALOVSKI, Yann BEAUXIS, Suzanne DIKKER, Guillaume DUMAS  
New contributors: Marine Gautier MARTINS, Rémy RAMADOUR, Patrice FORTIN, Ghazaleh RANJBARAN, Quentin MOREAU, Caitriona DOUGLAS, Franck PORTEOUS, Jonas MAGO, Juan C. AVENDANO, Julie BONNAIRE

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

📊 [shiny/\*.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/app) — Shiny dashboards, install using `poetry install --extras shiny` (Patrice)

## Poetry Installation (Only for Developers and Adventurous Users)

To develop HyPyP, we recommend using [Poetry 2.x](https://python-poetry.org/). Follow these steps:

### 1. Install Poetry:

```bash
   pip install poetry
```

### 2. Clone the Repository:

```bash
git clone git@github.com:ppsp-team/HyPyP.git
cd HyPyP
```

### 3. Install Dependencies:

```bash
poetry install
```

Note: By default, dev dependencies (including JupyterLab) are not included in the main dependencies.
To install development dependencies, you can run:

```bash
poetry install --with dev
```

### 4. Launch Jupyter Lab to Run Notebooks:

Instead of entering a shell, launch Jupyter Lab directly within the Poetry environment:

```bash
poetry run jupyter lab
```

## Additional Setup for Poetry

- **Install Poetry Plugins:**

To have full functionality with Poetry, add the following plugins:

```bash
poetry self add poetry-plugin-shell
poetry self add poetry-plugin-export
```

- **VS Code Integration:**

To make the Poetry virtual environment available in VS Code, you might need to set Poetry to create in-project virtual environments. You can do this either by running:

```bash
poetry config virtualenvs.in-project true
```

or by adding the following line to your .bashrc or .zshrc:

```bash
poetry config virtualenvs.in-project true
```

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
