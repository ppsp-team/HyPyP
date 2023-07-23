# HyPyP 🐍〰️🐍

The **Hy**perscanning **Py**thon **P**ipeline

[![PyPI version shields.io](https://img.shields.io/pypi/v/hypyp.svg)](https://pypi.org/project/HyPyP/) [![CI](https://github.com/ppsp-team/HyPyP/actions/workflows/Build.yml/badge.svg)](https://github.com/ppsp-team/HyPyP/actions/workflows/Build.yml) <a href="https://hypyp.readthedocs.io"><img src="https://readthedocs.org/projects/hypyp/badge/?version=latest"></a> [![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Mattermost](https://img.shields.io/discord/1065810348944408616?color=blue)](https://discord.gg/zYzjeGj7D6)

⚠️ This software is in beta and thus should be considered with caution. While we have done our best to test all the functionalities, there is no guarantee that the pipeline is entirely bug-free. 

📖 See our [paper](https://academic.oup.com/scan/advance-article/doi/10.1093/scan/nsaa141/5919711) for more explanation and our plan for upcoming functionalities (aka Roadmap).

🤝 If you want to help you can submit bugs and suggestions of enhancements in our Github [Issues section](https://github.com/ppsp-team/HyPyP/issues).

🤓 For the motivated contributors, you can even help directly in the developpment of HyPyP. You will need to install [Poetry](https://python-poetry.org/) (see section below).

## Contributors
Original authors: Florence BRUN, Anaël AYROLLES, Phoebe CHEN, Amir DJALOVSKI, Yann BEAUXIS, Suzanne DIKKER, Guillaume DUMAS
New contributors: Ghazaleh RANJBARAN, Quentin MOREAU, Caitriona DOUGLAS, Franck PORTEOUS, Jonas MAGO, Juan C. AVENDANO, Julie BONNAIRE

## Installation

```
pip install HyPyP
```

## Documentation

HyPyP documentation of all the API functions is available online at [hypyp.readthedocs.io](https://hypyp.readthedocs.io/)

For getting started with HyPyP, we have designed a little walkthrough: [getting_started.ipynb](https://github.com/ppsp-team/HyPyP/blob/master/tutorial/getting_started.ipynb)

## Core API

🛠 [io.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/io.py) — Loaders (Florence, Anaël, Ghazaleh, Franck, Jonas, Guillaume)

🧰 [utils.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/utils.py) — Basic tools (Amir, Florence, Guilaume)

⚙️ [prep.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/prep.py) — Preprocessing (ICA & AutoReject) (Anaël, Florence, Guillaume)

🔠 [analyses.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/analyses.py) — Power spectral density and wide choice of connectivity measures (Phoebe, Suzanne, Florence, Ghazaleh, Juan, Guillaume)

📈 [stats.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/stats.py) — Statistics (permutations & cluster statistics) (Florence, Guillaume)

🧠 [viz.py](https://github.com/ppsp-team/HyPyP/blob/master/hypyp/viz.py) — Inter-brain visualization (Anaël, Amir, Florence, Guillaume)

🎓 [Tutorials](https://github.com/ppsp-team/HyPyP/tree/master/tutorial) - Examples & documentation (Anaël, Florence, Yann, Ghazaleh, Caitriona, Guillaume)

## Poetry installation (only for developpers and adventurous users)

Step 1: ```pip install poetry```

Step 2: ```git clone git@github.com:ppsp-team/HyPyP.git```

Step 3: ```cd HyPyP```

Step 4: ```poetry install```

Step 5: ```poetry shell```

You can now use ```jupyter notebook``` or ```ipython```!

⚠️ If you need to install a new dependency (not recommended), you have to use `poetry add THE_NAME_OF_THE_LIBRARY` instead of your usual package manager.
