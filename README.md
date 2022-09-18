# SpectralClustering
Python and C implementation of Spectral Clustering, using Python-C API.

You can skip the building part,everything you need in order to run the program is already in the main folder.

## How to build (with Linux)

1)Open Terminal from src

2)Run the following commands:
```bash
python3 setup.py build_ext --inplace

bash comp.sh
```
## How to Execute from C/Python interface:
1)Open Terminal from the main folder

2)Run the following commands:

To run C interface:
```bash
/spkmeans <goal> <input file name>
```
To run Python interface:

To install numpy
```bash
pip install numpy
```
Then
```bash
python3 spkmeans.py <K> <goal> <input file name>
```
Please choose goal and K as explained in Assignments/final_project.pdf at pages 5-6

Made by @NivZindorf && @asafyi
