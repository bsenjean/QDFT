import sys, os
import numpy as np
import math
import qiskit.quantum_info
import scipy
import subprocess
import QDFT
from qiskit import transpile
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import NumPyEigensolver
from qiskit.circuit.library import TwoLocal
from qiskit_aer import QasmSimulator
from qiskit.visualization import circuit_drawer
from qiskit.quantum_info import Statevector, Operator
import SOFT

working_directory = os.getenv('QDFT_DIR')
code_directory = 'QDFT_DIR'
# ===========================================================#
# =============== Initialization by the user ================#
# ===========================================================#

# Quantum Simulator
nshots = False
simulation = ["noiseless", "noisy"][
    0]  # noiseless doesn't mean without sampling noise ! Noisy means a real simulation of a quantum computer.
if simulation == "noiseless":
    backend = Aer.get_backend('statevector_simulator')
    blocking = False
    allowed_increase = None
elif simulation == "noisy":
    # device_backend   = FakeVigo()
    # device           = QasmSimulator.from_backend(device_backend)
    # noise_model      = NoiseModel.from_backend(device)
    import qiskit.providers.aer.noise as noise

    # Error probabilities
    prob_1 = 0.0001  # 1-qubit gate
    prob_2 = 0.001  # 2-qubit gate
    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)
    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'ry', 'rz', 'sx', 'rx'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz'])

    backend = QasmSimulator(method='statevector', noise_model=noise_model)
    blocking = True
    allowed_increase = 0.1
else:
    sys.exit("Simulation argument '{}' does not exist".format(simulation))

n_blocks = 2  # Number of block considered into the Hardware efficient ansatz
rotation_blocks = ['ry', 'rz', ['ry', 'rz'], ['rz', 'rx', 'rz']][
    0]  # rotation gates in the ansatz ### I noticed RyRz works best (3 states optimization for instance, but much longer time...)
entanglement = ['full', 'linear', 'circular', 'sca'][1]  # the way qubits are entangled in the ansatz
entanglement_blocks = ['cz', 'cx'][
    1]  # entanglement gate in the ansatz ## I noticed cz works best (3 states optimization)

# SCF convergence criteria:
E_conv = 1e-5
D_conv = 1e-4
SCF_maxiter = 50
slope_SCF = 5  # set to None is not used.

# Classical optimizer criteria:
basinhopping = False
niter_hopping = 10  # parameter used for basinhopping
opt_method = ['L-BFGS-B', 'SLSQP', 'SPSA'][0]  # classical optimizer
opt_maxiter = 1000  # number of iterations of the classical optimizer
ftol = 2.220446049250313e-9
gtol = 1e-5
resampling = 2  # used for SPSA, see documentation.
slope_SPSA = 25  # set to None is not used.

# System:
n_qubits = 2
n_sites = 2 ** n_qubits
n_elec = 2
n_occ = n_elec // 2
t = 1.
# U_list         = [0.0001,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
U_list = [10.0]
potential = ["uniform", "random", "ABAB", "power", "decreasing"][4]

# SA weights:
weights_choice = ["equi", "decreasing", "equi_periodic", "equi_antiperiodic"][1]
automatic_weight_choice = False

# ===========================================================#
# =============== End of the initialization =================#
# ===========================================================#

if automatic_weight_choice:
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
    sys.exit("The choice for the weights-attribution is not defined.")

# ===========================================================#
# ============= External (physical) potential ===============#
# ===========================================================#

v = np.full((n_sites), 1. * n_elec / (1. * n_sites))  # uniform potential
if potential == "uniform":
    pass
elif potential == "random":
    for i in range(n_sites): v[i] = random.uniform(-1, 1)
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
    sys.exit("The potential is not defined. Program terminated.")

# ===========================================================#
# ================ CIRCUIT IMPLEMENTATION ===================#
# ===========================================================#

# Preparation of orthogonal initial states via different initial circuits (need n_occ of them !)
initial_circuits = []
for i in range(n_occ): initial_circuits += [QuantumCircuit(n_qubits)]

for state in range(n_occ):  # binarystring representation of the integer
    for i in QDFT.list_of_ones(state, n_qubits):
        initial_circuits[state].x(i)

circuits = [TwoLocal(n_qubits, rotation_blocks, entanglement_blocks, entanglement, n_blocks, insert_barriers=True,
                     initial_state=initial_circuits[state]) for state in range(n_occ)]
n_param = circuits[0].num_parameters
param_values = np.zeros(n_param)
print(circuits[0].decompose())
circuit_drawer(circuits[0].decompose(), scale=None, filename="circuit", style=None, output="latex_source",
               interactive=True, plot_barriers=True, reverse_bits=False, justify=None, vertical_compression='medium',
               idle_wires=True, with_layout=True, fold=None, ax=None, initial_state=False, cregbundle=True)
# ========================================================================#
# ============= START THE ALGORITHM FOR DIFFERENT U VALUES ===============#
# ========================================================================#
for U in U_list:

    print("*" * 50)
    print("*" + " " * 18 + "U = {:8.3f}".format(U) + " " * 18 + "*")
    print("*" * 50)

    output_file = working_directory + "examples/results/L{}_N{}_U{}_{}_{}_nshots{}_layer{}_maxiter{}_resampling{}_slopeSPSA{}_slopeSCF{}_{}.dat".format(
        n_sites, n_elec, U, opt_method, potential, nshots, n_blocks, opt_maxiter, resampling, slope_SPSA, slope_SCF,
        simulation)
    with open(output_file, 'w+') as f:
        f.write('')

    # perform DFT to compute fidelities wrt exact KS orbitals
    KS_orbs, E_DFT, density_exact = SOFT.run_SOFT_Hubbard(n_sites, n_elec, U, t, v, SCF_maxiter, code_directory,
                                                          output_file)
    proba_states_exact = [np.abs(KS_orbs[:, state]) ** 2 for state in range(n_occ)]

    density = np.full((n_sites), 1. * n_elec / (1. * n_sites))
    mix_cst = 0.4
    Etot = 0
    Delta_E = 1e8

    with open(output_file, 'a') as f:
        f.write(str(circuits[0].decompose()) + "\n")

    last_energies = []
    for SCF_ITER in range(1, SCF_maxiter + 1):

        print("iteration ", SCF_ITER)

        # Compute the Hxc BALDA potential:
        # subprocess.check_call("echo " + str(U) + " " + str(t) + " | beta_and_derivatives",shell=True, cwd = code_directory)
        with open(code_directory + "beta_dbetadU.dat", "r") as f:
            line = f.read()
            beta = float(line.split()[0])
            dbeta_dU = float(line.split()[1])
            f.close()
        deHxc_dn = SOFT.generate_potential(n_sites, U, t, density, beta, dbeta_dU)[1]

        # Build the reference non-interacting 1D Hubbard Hamiltonian with periodic or antiperiodic conditions.
        h_KS = SOFT.generate_hamiltonian(n_sites, n_elec, t, v + deHxc_dn)
        H_qubit = QDFT.transformation_Hmatrix_Hqubit(h_KS, n_qubits)[0]
        solver = NumPyEigensolver(k=2 ** n_qubits)
        result = solver.compute_eigenvalues(H_qubit)
        eigvals = result.eigenvalues
        eigvecs = np.array(qiskit.quantum_info.Operator(result.eigenstates).transpose())

        with open(output_file, 'a') as f:
            f.write("varepsilon from diag: {}".format(eigvals[:n_occ]) + "\n")
            f.write("weights: {}".format(weights) + "\n")

        if opt_method == "L-BFGS-B": opt_options = {'maxiter': opt_maxiter, 'ftol': ftol, 'gtol': gtol}
        if opt_method == "SLSQP": opt_options = {'maxiter': opt_maxiter, 'ftol': ftol}
        if opt_method != "SPSA":
            if not basinhopping:
                f_min = scipy.optimize.minimize(QDFT.cost_function_energy,
                                                x0=(param_values),
                                                args=(circuits,
                                                      H_qubit,
                                                      weights,
                                                      simulation,
                                                      nshots,
                                                      backend,
                                                      False),
                                                method=opt_method,
                                                options=opt_options)
            else:
                f_min = scipy.optimize.basinhopping(QDFT.cost_function_energy,
                                                    x0=(param_values),
                                                    niter=niter_hopping,
                                                    # niter steps of basinhopping, and niter+1 optimizer iterations for each step.
                                                    minimizer_kwargs={'method': opt_method,
                                                                      'args': (circuits,
                                                                               H_qubit,
                                                                               weights,
                                                                               simulation,
                                                                               nshots,
                                                                               backend,
                                                                               False)})
            param_values = f_min['x']
            E_SA = f_min['fun']
        else:
            spsa = SPSA(maxiter=opt_maxiter, blocking=blocking, allowed_increase=allowed_increase, last_avg=slope_SPSA,
                        resamplings=resampling)
            cost_function = SPSA.wrap_function(QDFT.cost_function_energy,
                                               (circuits,
                                                H_qubit,
                                                weights,
                                                simulation,
                                                nshots,
                                                backend,
                                                False))
            result = spsa.minimize(cost_function, x0=param_values)
            param_values = result.x
            E_SA = result.fun  # if last_avg is not 1, it returns the callable function with the last_avg param_values as input ! This seems generally good and better than taking the mean of the last_avg function calls.
            stddev = spsa.estimate_stddev(cost_function, initial_point=param_values)

        bounds = [circuits[state].assign_parameters(param_values) for state in range(n_occ)]
        if nshots is not False:
            energies = [sampled_expectation_value(bounds[state], H_qubit, backend, simulation, nshots=nshots) for state
                        in range(n_occ)]
        else:
            opp = Operator(H_qubit)
            energies = [np.real(Statevector.from_instruction(bounds[state]).expectation_value(opp)) for state in
                        range(n_occ)]

        with open(output_file, 'a') as f:
            f.write("SA ENERGY: {}".format(E_SA) + "\n")

        # SOFT vs DFT: no need to have the full state, just the probability density !
        if simulation == "noiseless":
            bounds = [circuits[state].assign_parameters(param_values) for state in range(n_occ)]
            states = [np.array(Statevector(bounds[state])) for state in range(n_occ)]
            if nshots is not False:
                # states = [sampled_state(bounds[state],n_qubits,simulation,backend=backend,nshots=nshots) for state in range(n_occ)] # Grover... actually not needed.
                proba_states = [np.random.multinomial(nshots, (np.abs(states[state]) ** 2)) / nshots for state in
                                range(n_occ)]
            else:  # state vector simulation
                proba_states = [np.abs(states[state]) ** 2 for state in range(n_occ)]
        elif simulation == "noisy":
            states = []
            proba_states = []
            for state in range(n_occ):
                bound = circuits[state].bind_parameters(param_values)
                bound.save_statevector()
                # Transpile for simulator
                bound = transpile(bound, backend)
                result = backend.run(bound).result()
                state = result.get_statevector(bound)
                proba_state = np.random.multinomial(nshots, (np.abs(state) ** 2)) / nshots
                states.append(state)
                proba_states.append(proba_state)
        else:
            sys.exit("Simulation argument '{}' does not exist".format(simulation))

        # Compute the density
        density_new = np.zeros((n_sites))
        for orbital in range(n_sites):
            for occ in range(n_occ):
                density_new[orbital] += 2 * proba_states[occ][orbital]

        # Compute the Hxc energy contribution with the new density
        eHxc = SOFT.generate_potential(n_sites, U, t, density_new, beta, dbeta_dU)[0]

        # Compute the total KSDFT energy
        sum_occ_eKS = 0
        sum_eHxc = 0
        sum_deHxc_dn = 0
        for i in range(n_occ):   sum_occ_eKS += 2. * energies[i]
        for i in range(n_sites): sum_eHxc += eHxc[i]
        for i in range(n_sites): sum_deHxc_dn += deHxc_dn[i] * density_new[i]
        Etot_new = sum_occ_eKS + sum_eHxc - sum_deHxc_dn
        Delta_E = abs(Etot_new - Etot)
        Etot = Etot_new
        # Compute the norm of the change of density between the iterations.
        normocc = np.linalg.norm(density - density_new)
        # Compute the norm of the change of density between the iteration and the exact one:
        normocc_exact = np.linalg.norm(density - density_exact)

        # Check convergence and increase mix_cst if reached
        # if (Delta_E <= SCF_energy_tol and normocc <= SCF_density_tol): mix_cst += 0.2
        # Kind of DIIS algorithm: reduce the change in density to avoid convergence issues
        density = (1 - mix_cst) * density + mix_cst * density_new

        fidelities = np.array([abs(np.conj(states[i]).T @ KS_orbs[:, i]) ** 2 for i in range(n_occ)])
        if isinstance(fidelities, int): fidelities = [fidelities]
        if nshots is not False:
            fidelities_proba = np.array([np.linalg.norm(proba_states[i] - proba_states_exact[i]) for i in range(n_occ)])
            if isinstance(fidelities_proba, int): fidelities_proba = [fidelities_proba]
        # Print results
        with open(output_file, 'a') as f:
            f.write("*" * 10 + " ITERATION {:3d} ".format(SCF_ITER) + "*" * 10 + "\n")
            f.write("Energy (hartree) : {:16.8f}".format(Etot) + "\n")
            f.write("Occupied KS nrj  : {}".format(energies) + "\n")
            f.write("New    density   : {}".format(density_new) + "\n")
            f.write("Damped density   : {}".format(density) + "\n")
            f.write("Fidelity wrt ED  : {}".format(
                [abs(np.conj(states[i]).T @ eigvecs[:, i]) ** 2 for i in range(n_occ)]) + "\n")
            f.write("Fidelity wrt KS  : {}".format(fidelities) + "\n")
            if nshots is not False: f.write("DiffNorm proba   : {}".format(fidelities_proba) + "\n")
            f.write("Delta E iter     : {:16.8f}".format(Delta_E) + "\n")
            f.write("Delta E DFTexact : {:16.8f}".format(Etot - E_DFT) + "\n")
            f.write("Norm Delta_occ   : {:16.8f}".format(normocc) + "\n")
            f.write("Norm Delta_occ_exact: {:16.8f}".format(normocc_exact) + "\n")

        if slope_SCF is not None:
            last_energies.append(Etot)

            if len(last_energies) > slope_SCF:
                last_energies = last_energies[-slope_SCF:]
                pp = np.polyfit(range(slope_SCF), last_energies, 1)
                slope = pp[0]
                origin_intersection = pp[1]
                with open(output_file, 'a') as f:
                    f.write("Slope SCF        : {:16.8f}".format(slope) + "\n")
                    f.write("Origin Inters.   : {:16.8f}\n".format(origin_intersection) + "\n")

                if abs(slope) < 5e-4:
                    Delta_E = 0.
                    normocc = 0.

        if ((Delta_E < E_conv) and (normocc < D_conv)) or SCF_ITER == SCF_maxiter:

            rel_error_EVQE_EDFT = abs((Etot - E_DFT) / E_DFT)

            with open(output_file, 'a') as f:
                if SCF_ITER == SCF_maxiter: f.write("*" * 10 + " FAILURE " + "*" * 10 + "\n")
                if (Delta_E < E_conv) and (normocc < D_conv): f.write("*" * 10 + " SUCCESS " + "*" * 10 + "\n")
                f.write("Iteration        : {:16d}".format(SCF_ITER) + "\n")
                f.write("States fidelity  : {}".format(fidelities) + "\n")
                if nshots is not False: f.write("DiffNorm proba   : {}".format(fidelities_proba) + "\n")
                f.write("Rel Err EVQE/EDFT: {:16.8f}".format(rel_error_EVQE_EDFT) + "\n")
                f.write("DFT energy exact : {:16.8f}".format(E_DFT) + "\n")
                f.write("DFT energy SAVQE : {:16.8f}".format(Etot) + "\n")
                if opt_method == "SPSA": f.write("stddev SPSA      : {:16.8f}".format(stddev) + "\n")
            break
