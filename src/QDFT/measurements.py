import numpy as np
import math
from qiskit.opflow import StateFn
from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit.circuit.library import MCMT
import sys

def list_of_ones(computational_basis_state: int, n_qubits):
    """
    Indices of ones in the binary expansion of an integer in big endian
    order. e.g. 010110 -> [1, 3, 4] (which is the reverse of the qubit ordering...)
    """

    bitstring = format(computational_basis_state, 'b').zfill(n_qubits)

    return [abs(j-n_qubits+1) for j in range(len(bitstring)) if bitstring[j] == '1']

def cost_function_energy(param_values,circuits,H_qubit,weights,simulation,nshots=False,backend=False,output=False):

    nstates = len(weights)
    E_SA = 0.

    bounds = [circuits[state].bind_parameters(param_values) for state in range(nstates)]
    if nshots is not False:
       energies = [sampled_expectation_value(bounds[state],H_qubit,backend,simulation,nshots=nshots) for state in range(nstates)]
    else:
       energies = [np.real((StateFn(H_qubit, is_measurement=True) @ StateFn(bounds[state])).eval()) for state in range(nstates)]

    # Compute the state-averaged energy
    E_SA = 0.
    for i in range(nstates): E_SA += energies[i] * weights[i]

    if output is not False:
      with open(output,'a') as f: f.write('{}\n'.format(E_SA))

    return E_SA

def Grover_diffusion_circuit(circuit,n_qubits):
    """ Construct the Grover diffusion operator circuit.
    circuit: QuantumCircuit object
    n_qubits: number of qubits

    Returns: the circuit with the added Grover diffusion circuit.
    """

    circuit_Grover = circuit.copy()
    for i in range(n_qubits):
      circuit_Grover.h(i)
      circuit_Grover.x(i)
    circuit_Grover += MCMT('z',n_qubits-1,1)
    for i in range(n_qubits):
      circuit_Grover.x(i)
      circuit_Grover.h(i)

    return circuit_Grover

def sampled_state(original_circuit,n_qubits,simulation,backend=False,nshots=1024):

    print("** WARNING ** This function has not been tested yet.")
    circuit = original_circuit.copy()
    circuit_Grover = Grover_diffusion_circuit(original_circuit,n_qubits)
    if simulation == "noiseless":
      state = np.array(Statevector(circuit))
      state_Grover = np.array(Statevector(circuit_Grover))
      # Compute the proba of getting a given bitstring, based on a given distribution defined by the number of shots.
      proba_state_array = np.random.multinomial(nshots,(np.abs(state)**2).real)
      proba_state_Grover_array = np.random.multinomial(nshots,(np.abs(state_Grover)**2).real)
      proba_state = {}
      proba_state_Grover = {}
      for indice in range(len(proba_state_array)):
          if proba_state_array[indice] >= 1:
             proba_state[str(indice)] = proba_state_array[indice]/nshots
          if proba_state_Grover_array[indice] >= 1:
             proba_state_Grover[str(indice)] = proba_state_Grover_array[indice]/nshots

    elif simulation == "noisy":
      circuit.measure_all()
      circuit_Grover.measure_all()
      # Transpile for simulator
      circuit = transpile(circuit, backend)
      circuit_Grover = transpile(circuit_Grover, backend)
      # Run and get counts
      result = backend.run(circuit,shots=nshots).result()
      counts = result.get_counts(circuit) # The string is little-endian (cr[0] on the right hand side).
      # transform the dictionary with binary to integer:
      proba_state = {}
      for item in counts.items():
        proba_state[str(int(item[0],2))] = item[1]/nshots
      # Run and get counts
      result = backend.run(circuit_Grover,shots=nshots).result()
      counts = result.get_counts(circuit_Grover) # The string is little-endian (cr[0] on the right hand side).
      # transform the dictionary with binary to integer:
      proba_state_Grover = {}
      for item in counts.items():
        proba_state_Grover[str(int(item[0],2))] = item[1]/nshots

    else:
      sys.exit("Simulation argument '{}' does not exist".format(simulation))

    # combine dictionnaries to have the same keys:
    combine_dict = proba_state | proba_state_Grover
    for key in combine_dict.keys():
      if key not in proba_state: proba_state[key] = 0
      if key not in proba_state_Grover: proba_state_Grover[key] = 0

    # Find the 4 different values of mean of the coefficients:
    mean_Grover = []
    for key in combine_dict:
      mean_Grover.append(abs(0.5*( np.sqrt(proba_state[key]) + np.sqrt(proba_state_Grover[key]))))
      mean_Grover.append(abs(0.5*(-np.sqrt(proba_state[key]) - np.sqrt(proba_state_Grover[key]))))
      mean_Grover.append(abs(0.5*( np.sqrt(proba_state[key]) - np.sqrt(proba_state_Grover[key]))))
      mean_Grover.append(abs(0.5*(-np.sqrt(proba_state[key]) + np.sqrt(proba_state_Grover[key]))))

    # Example usage:
    around = math.floor(math.log(np.sqrt(nshots), 10)) + 1
    mean_Grover = np.around(mean_Grover,around).tolist()
    dict_of_counts = {value:mean_Grover.count(value) for value in mean_Grover}
    chosen_one = max(dict_of_counts, key=dict_of_counts.get)

    coefficients = []
    for i in range(2**n_qubits):
      key = str(i)
      if key in combine_dict:      
        coefficients.append(( proba_state[key] + 4*chosen_one**2 - proba_state_Grover[key]) / (4*chosen_one))
      else:
        coefficients.append(0)

    return coefficients/np.linalg.norm(coefficients)

def sampled_expectation_value(original_circuit,operator,backend,simulation,nshots=1024):
    rot_dic = { 'X' : lambda qubit : circuit.h(qubit),
                'Y' : lambda qubit : circuit.rx(np.pi/2., qubit)}

    try:
     nterms = len(operator)
    except:
     nterms = 1
    nqubits = operator.num_qubits
    nshots_per_pauli = int(nshots/nterms)
    expectation_value = 0
    if nterms == 1: operator = [operator]
    for i in range(nterms):
        circuit = original_circuit.copy()
        # Begining of the circuit fragment: we detect if Pauli op is X, Y or Z
        # and apply a rotation gate accordingly on the associated qubits
        # store the places where there is a X, Y or Z operators as well (all rotated to Z)
        list_Z = []
        for j in reversed(range(nqubits)): 
            # For some reason, if str(operator[i].to_pauli_op().primitive) --> ZII, we have
            # str(operator[i].to_pauli_op().primitive[0]) = "I" and
            # str(operator[i].to_pauli_op().primitive[2]) = "Z".............. 
            # This messed up so much with my brain.
            if str(operator[i].to_pauli_op().primitive[j]) == 'I':
                continue
            elif str(operator[i].to_pauli_op().primitive[j]) == 'Z':
                list_Z.append(j)
            else:
                rot_dic[str(operator[i].to_pauli_op().primitive[j])](j)
                list_Z.append(j)

        # Get the final state of the rotated circuit:
        if simulation == "noiseless":
          state = np.array(Statevector(circuit))
          # Compute the proba of getting a given bitstring, based on a given distribution defined by the number of shots.
          proba_computational_basis_array = np.random.multinomial(nshots_per_pauli,(np.abs(state)**2).real)
          proba_computational_basis = {}
          for indice in range(len(proba_computational_basis_array)):
              if proba_computational_basis_array[indice] >= 1:
                 proba_computational_basis[str(indice)] = proba_computational_basis_array[indice]/nshots_per_pauli

        elif simulation == "noisy": # if noisy, we cannot get the state directly, we have to combine the measurements to extract the energy.
          circuit.measure_all()
          # Transpile for simulator
          circuit = transpile(circuit, backend)
          # Run and get counts
          result = backend.run(circuit,shots=nshots_per_pauli).result()
          counts = result.get_counts(circuit) # The string is little-endian (cr[0] on the right hand side).
          # transform the dictionary with binary to integer:
          proba_computational_basis = {}
          for item in counts.items():
            proba_computational_basis[str(int(item[0],2))] = item[1]/nshots_per_pauli

        else:
          sys.exit("Simulation argument '{}' does not exist".format(simulation))

        # Compute the energy by combining the proba:
        for integer_bitstring in range(2**nqubits):
          if str(integer_bitstring) in proba_computational_basis:
            phase = 1
            # Determine the phase:
            for qubit in list_Z:
              if qubit in list_of_ones(integer_bitstring,nqubits):
                phase *= -1

            # Determine the expectation value:
            expectation_value += phase * proba_computational_basis[str(integer_bitstring)] * operator[i].to_pauli_op().coeff

    return expectation_value
