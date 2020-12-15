#!/bin/bash

: '
This short bash/shell script will run test scripts found in the tests folder
'

cd tests/chebyshev
python test_bvp.py
python test_interpolation.py

cd ../legendre
python test_associated_legendre.py

cd ../spherical-harmonic
python test_spherical_harmonic.py
python test_noise.py

cd ../../
