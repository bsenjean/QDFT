#!/usr/bin/env python3

from .geometry import Hchain_geometry

from .operators import (
  decompose_operator_to_pauli_list,
  transformation_Hmatrix_Hqubit,
  sz_operator,
  s2_operator)


from .measurements import (
  list_of_ones,
  circuits,
  cost_function_energy,
  Grover_diffusion_circuit,
  sampled_state,
  sampled_expectation_value)




def simulation(simulation: str, weights_choice: str, nshots: str or int, Hubbard: bool,  n_elec: int, n_occ: int, n_qubits: int, p1: float =0.0001, p2: float = 0.001, potential: str = "uniform"):
    '''Inizializing the simulations'''

    import numpy as np
    from qiskit_aer import QasmSimulator, Aer
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeVigo
    import sys
    n_sites = 2 ** n_qubits
    v = np.ndarray
    if simulation == "noisy":
        if type(nshots) != int:
            sys.exit("QDFT error: You are running a calculation in a Noisy QC. nshots can't be False.")
        else:
            if Hubbard != True:
                device_backend = FakeVigo()
                device = QasmSimulator.from_backend(device_backend)
                noise_model = NoiseModel.from_backend(device)
            else:
                ## Error probabilities
                prob_1: float = p1  # 1-qubit gate
                prob_2: float = p2  # 2-qubit gate
                print("Note: The Error probabilities values are:", prob_1, prob_2)
                ## Depolarizing quantum errors
                import qiskit_aer.noise as noise
                error_1 = noise.depolarizing_error(prob_1, 1)
                error_2 = noise.depolarizing_error(prob_2, 2)
                ## Add errors to noise model
                noise_model = NoiseModel()
                noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'sx', 'x'])
                noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

            backend = QasmSimulator(method='statevector', noise_model=noise_model)
            blocking = True
            allowed_increase = 0.1
    elif simulation == "noiseless":
        # (print("Error probability values have no effect on noiseless calculations"))
        backend = Aer.get_backend('statevector_simulator')
        blocking = False
        allowed_increase = None
    else:
        sys.exit("Simulation argument '{}' does not exist".format(simulation))

    '''Define weight'''
    if weights_choice == "auto":
        if (n_elec // 2) % 2 == 1: weights_choice = "equi_periodic"
        if (n_elec // 2) % 2 == 0: weights_choice = "equi_antiperiodic"
    if weights_choice == "equi":
        weights = [1. / n_occ for i in range(n_occ)]  # should be n_occ of them
    elif weights_choice == "decreasing":
        weights = [(1. + i) / (n_occ * (n_occ + 1.) / 2) for i in reversed(range(n_occ))]  # should be n_occ of them
    elif weights_choice == "equi_antiperiodic":
        weights = []
        for i in reversed(range(n_occ // 2)):
            weights.append((1. + i) / ((n_occ // 2) * ((n_occ // 2) + 1.)))
            weights.append((1. + i) / ((n_occ // 2) * ((n_occ // 2) + 1.)))
    elif weights_choice == "equi_periodic":
        first_weight = (1. + (n_occ - 1)) / (n_occ * (n_occ + 1.) / 2)
        weights = [first_weight]
        for i in reversed(range(n_occ // 2)):
            weights.append((1. + i) / ((n_occ // 2) * ((n_occ // 2) + 1.)) - first_weight / (n_occ - 1))
            weights.append((1. + i) / ((n_occ // 2) * ((n_occ // 2) + 1.)) - first_weight / (n_occ - 1))
    else:
        sys.exit("The choice for the weights-attribution is not defined. Usage of the following keywords are allowed: equi, decreasing, equi_periodic, equi_antiperiodic, auto")

    '''Define potential'''
    if Hubbard != True:
        pass
    else:
        v = np.full((n_sites), 1. * n_elec / (1. * n_sites))  # uniform potential
        if potential == "uniform":
            print("Default potential is used: potential = uniform")
            pass
        elif potential == "random":
            for i in range(n_sites): v[i] = np.random.uniform(-1, 1)
        elif potential == "ABAB":
            for i in range(n_sites // 2):
                v[2 * i] = -1.
                v[2 * i + 1] = +1.
        elif potential == "power":  # See GAO XIANLONG et al. PHYSICAL REVIEW B 73, 165120 (2006)
            l = 2
            const_V = 0.006
            for i in range(n_sites):
                v[i] = const_V * (i - n_sites / 2) ** l
        elif potential == "decreasing":
            for i in reversed(range(n_sites)):
                v[i] = 0.1 * i
        else:
            sys.exit("The potential is not correctly defined. Usage of the following keywords are allowed: uniform, random, ABAB, power, decreasing")

    if not Hubbard:
        return backend, blocking, allowed_increase, weights
    else:
        return backend, blocking, allowed_increase, weights, potential, n_sites, v


