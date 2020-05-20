import os
from setuptools import setup

setup(
    name="HyPyP",
    version="0.1.2",
    author="Anaël AYROLLLES, Florence BRUN, Phoebe CHEN, Amir DJALOVSKI, Yann BEAUXIS, Suzanne DIKKER, Guillaume DUMAS",
    author_email="anael.ayrollles@pasteur.fr, florence.brun@pasteur.fr, phoebe.chen1117@gmail.com, amir.djv@gmail.com, dev@yannbeauxis.net, suzanne.dikker@nyu.edu, guillaume.dumas@pasteur.fr",
    maintainer="Guillaume DUMAS",
    maintainer_email="guillaume.dumas@centraliens.net",
    description="The Hyperscanning Python Pipeline.",
    license="BSD",
    keywords="hyperscanning, EEG, MEG, pipeline, statistics, visualization",
    url="https://github.com/GHFC/HyPyP",
    packages=['hypyp'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
    ],
)
