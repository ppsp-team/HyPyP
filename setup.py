import os
from setuptools import setup

setup(
    name="HyPyP",
    version="0.1",
    author="Florence BRUN, Anaël AYROLLLES, Phoebe CHEN, Amir DJALOVSKI, Suzanne DIKKER, Guillaume DUMAS",
    author_email="florence.brun@pasteur.fr, anael.ayrollles@pasteur.fr, phoebe.chen1117@gmail.com, amir.djv@gmail.com, suzanne.dikker@nyu.edu, guillaume.dumas@pasteur.fr",
    maintainer="Guillaume DUMAS",
    maintainer_email="guillaume.dumas@pasteur.fr",
    description="The Hyperscanning Python Pipeline.",
    license="BSD",
    keywords="hyperscanning EEG",
    url="https://github.com/GHFC/HyPyP",
    packages=['hypyp'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
    ],
)
