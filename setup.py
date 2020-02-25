import os
from setuptools import setup

setup(
    name = "HyPyR",
    version = "0.1",
    author = "Florence BRUN, Anaël AYROLLLES, Guillaume DUMAS",
    author_email = "florence.brun@pasteur.fr, anael.ayrollles@pasteur.fr, guillaume.dumas@pasteur.fr",
    maintainer = "Guillaume DUMAS",
    maintainer_email = "guillaume.dumas@pasteur.fr",
    description = "A hyperscanning toolkit in Python.",
    license = "BSD",
    keywords = "hyperscanning EEG",
    url = "https://github.com/GHFC/HyPyR",
    packages=['hypyr'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
    ],
)
