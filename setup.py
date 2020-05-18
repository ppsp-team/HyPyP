import os
from setuptools import setup

setup(
    name="HyPyP",
    version="0.2",
    author="Florence BRUN, Anaël AYROLLLES, Phoebe CHEN, Amir DJALOVSKI, Yann BEAUXIS, Suzanne DIKKER, Guillaume DUMAS",
    author_email="florence.brun@pasteur.fr, anael.ayrollles@pasteur.fr, phoebe.chen1117@gmail.com, amir.djv@gmail.com, dev@yannbeauxis.net, suzanne.dikker@nyu.edu, guillaume.dumas@pasteur.fr",
    maintainer="Guillaume DUMAS",
    maintainer_email="guillaume.dumas@centraliens.net",
    description="The Hyperscanning Python Pipeline.",
    license="BSD",
    keywords="hyperscanning EEG",
    url="https://github.com/GHFC/HyPyP",
    packages=['hypyp'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
    ],
)
