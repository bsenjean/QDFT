# QDFT
Quantum Density Functional Theory: a quantum algorithm to solve the Kohn-Sham self-consistent equations.

# Installation

export QDFT_DIR=`pwd`

pip install -e .

# execution

cd examples

python3 DFT_Hchain.py

python3 QDFT_Hchain.py

python3 QDFT_Hubbard.py



For Hubbard, one requires to clone and install git@github.com:bsenjean/SOFT.git
