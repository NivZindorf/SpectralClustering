import numpy
from setuptools import setup, find_packages, Extension

setup(name="mykmeanssp",
        version="1.0.0",
        description="Python interface for the C library functions",
        author="Niv Zindorf",
        author_email="nivzindorf97@gmail.com",
        install_requires = ['invoke'],
        packages= find_packages(),
        ext_modules=[Extension("mykmeanssp",
        ["spkmeansmodule.c", "spkmeans.c"],
        include_dirs=[numpy.get_include()]
        )])