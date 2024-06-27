# QDFT - V 0.1.0
Quantum Density Functional Theory: a quantum algorithm to solve the Kohn-Sham self-consistent equations.


# Whats new
The Code is upgraded to adopt QISKIT V 1.1.0.

Few modulus are removed from qiskit. (Check https://docs.quantum.ibm.com/api/migration-guides/qiskit-1.0 for more details).

The changes in the current version of QDFT are as follows,

| version 0.0.0                   | Version 0.1.0                     | 
|---------------------------------|-----------------------------------|
| qiskit.Aer                      | qiskit_aer.Aer                    |
| qiskit.algorithms               | qiskit_algorithms                 |
| qiskit.opflow.StateFn           | qiskit.quantum_info.Statevector   |
| qiskit.opflow.I,X,Y             | qiskit.quantum_info.Pauli         |
| qiskit.circuits.bind_parameters | qiskit.circuits.assign_parameters |

- Pauli operations are carried out with qiskit.quantum_info.Operators.
- Note that the expectation values in the previous version is calculated using StateFn whereas, in the current release it is substituted with qiskit.quantum_info.Statevector. (We tried qiskit.primitives.Estimator as an alternative but it didn't workout well)
- New function called decompose_operator_to_pauli_list is added in the operators to change operators into pauli_list.
- Major changes are carried out in Operators.py (fun: transformation_Hmatrix_Hqubit) and measurements.py (fun: cost_function_energy).

# Installation

export QDFT_DIR=`pwd`

pip install -e .

# execution

cd examples

python3 DFT_Hchain.py

python3 QDFT_Hchain.py

python3 QDFT_Hubbard.py

Prerequisites:

- PSI4 should be installed and exported to the python path.

- For Hubbard, one requires to clone and install git@github.com:bsenjean/SOFT.git
  (Note: Make sure you define working directory and code directory correctly.)
