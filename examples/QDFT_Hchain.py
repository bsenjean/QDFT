import sys, os
import numpy as np
import math
import scipy
import QDFT
# importing Qiskit
from qiskit import Aer, transpile
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms import NumPyEigensolver
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer import QasmSimulator, AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo
from qiskit.visualization import circuit_drawer
from qiskit.opflow import StateFn

sys.path.insert(1, os.path.abspath('/usr/local/psi4/lib/'))
import psi4

psi4.set_memory('100 GB')

working_directory = os.getenv('QDFT_DIR')

#===========================================================#
#=============== Initialization by the user ================#
#===========================================================#

# Quantum Simulator
nshots         = False
simulation     = ["noiseless","noisy"][0] # noiseless doesn't mean without sampling noise ! Noisy means a real simulation of a quantum computer.
if simulation == "noiseless":
  backend          = Aer.get_backend('statevector_simulator')
  blocking         = False
  allowed_increase = None
elif simulation == "noisy":
  device_backend   = FakeVigo()
  device           = QasmSimulator.from_backend(device_backend)
  noise_model      = NoiseModel.from_backend(device)
  #import qiskit.providers.aer.noise as noise
  ## Error probabilities
  #prob_1 = 0.001  # 1-qubit gate
  #prob_2 = 0.1   # 2-qubit gate
  ## Depolarizing quantum errors
  #error_1 = noise.depolarizing_error(prob_1, 1)
  #error_2 = noise.depolarizing_error(prob_2, 2)
  ## Add errors to noise model
  #noise_model = noise.NoiseModel()
  #noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'sx', 'x'])
  #noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
  #backend        = QasmSimulator(method='statevector',noise_model=noise_model)
  backend = QasmSimulator(method='statevector', noise_model=noise_model)
  blocking         = True
  allowed_increase = 0.1
else:
  sys.exit("Simulation argument '{}' does not exist".format(simulation))

n_blocks       = 2 # Number of block considered into the Hardware efficient ansatz
rotation_blocks= ['ry','rz',['ry','rz'],['rz','rx','rz']][0] # rotation gates in the ansatz ### I noticed RyRz works best (3 states optimization for instance, but much longer time...)
entanglement   = ['full','linear','circular','sca'][1] # the way qubits are entangled in the ansatz
entanglement_blocks = ['cz','cx'][1] # entanglement gate in the ansatz ## I noticed cz works best (3 states optimization)

# SCF convergence criteria:
E_conv         = 1e-5
D_conv         = 1e-4
SCF_maxiter    = 50
slope_SCF      = 5 # set to None is not used.

# Classical optimizer criteria:
basinhopping   = False
niter_hopping  = 10 # parameter used for basinhopping
opt_method     = ['L-BFGS-B','SLSQP','SPSA'][0]  # classical optimizer
opt_maxiter    = 1000 # number of iterations of the classical optimizer
ftol           = 1e-9
gtol           = 1e-6
take_min_opt   = False # take the minimal energy point of the minimization. It is known not to be a good idea...
resampling     = 2 # used for SPSA, see documentation.
slope_SPSA     = 25 # set to None is not used.

# DFT
functional     = "SVWN"
basis          = "sto-3g"

# Other options:
run_fci        = True

# System:
n_qubits       = 2
n_orbs         = 2**n_qubits # number of hydrogens
n_elec         = n_orbs
n_occ          = n_elec//2
#interdist_list = [0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
interdist_list = [1.2]

# SA weights:
weights_choice = ["equi","decreasing"][1]

#===========================================================#
#=============== End of the initialization =================#
#===========================================================#

if weights_choice == "equi":
  weights = [1./n_occ for i in range(n_occ)] # should be n_occ of them
elif weights_choice == "decreasing":
  weights = [(1.+i)/(n_occ*(n_occ+1.)/2) for i in reversed(range(n_occ))] # should be n_occ of them
else:
  sys.exit("The choice for the weights-attribution is not defined.")

#===========================================================#
#================ CIRCUIT IMPLEMENTATION ===================#
#===========================================================#

# Preparation of orthogonal initial states via different initial circuits (need n_occ of them !)
initial_circuits = []
for i in range(n_occ): initial_circuits += [QuantumCircuit(n_qubits)]

for state in range(n_occ): # binarystring representation of the integer
    for i in QDFT.list_of_ones(state,n_qubits):
        initial_circuits[state].x(i)

circuits    = [TwoLocal(n_qubits,rotation_blocks,entanglement_blocks,entanglement,n_blocks,insert_barriers=True,initial_state=initial_circuits[state]) for state in range(n_occ)]
n_param     = circuits[0].num_parameters
param_values= np.zeros(n_param)
print(circuits[0].decompose())
circuit_drawer(circuits[0].decompose(), scale=None, filename="circuit", style=None, output="latex_source", interactive=True, plot_barriers=True, reverse_bits=False, justify=None, vertical_compression='medium', idle_wires=True, with_layout=True, fold=None, ax=None, initial_state=False, cregbundle=True)

#========================================================================#
#============= START THE ALGORITHM FOR DIFFERENT U VALUES ===============#
#========================================================================#
for R in interdist_list:

    psi4.core.clean()
    # Define the geometry
    psi4.core.set_output_file(working_directory + "examples/results/H{}_R{}_{}_{}_Psi4.dat".format(n_orbs,R,basis,functional),True)
    psi4.geometry(QDFT.Hchain_geometry("linear",n_orbs,R))

    psi4.set_options({'basis': basis, 'scf_type': 'pk'})#, 'print': 5, 'debug': 1})
    dft_e, dft_wfn = psi4.energy(functional, return_wfn=True)
    # Hcore and J (Coulomb) matrices:
    Hcore = dft_wfn.H().clone()
    E_nuc = dft_wfn.get_energies('Nuclear')
    # Density matrix in the AO basis:
    D_AO_DFT = dft_wfn.Da().clone()
    # Overlap matrix in the AO basis:
    S_AO = dft_wfn.S().np
    # KS-MO coefficient matrix in the AO basis:
    C_AO = dft_wfn.Ca().np
    # Compute the inverse square root of the overlap matrix S
    S_eigval, S_eigvec = np.linalg.eigh(S_AO)
    S_sqrt_inv = S_eigvec @ np.diag((S_eigval)**(-1./2.)) @ S_eigvec.T
    C_transformation = np.linalg.inv(S_sqrt_inv)
    C_OAO = C_transformation @ C_AO
    proba_states_exact = [np.abs(C_OAO[:,state])**2 for state in range(n_occ)]

    # Construct the SAD Guess for the initial density D_AO
    psi4.core.prepare_options_for_module("SCF")
    sad_basis_list = psi4.core.BasisSet.build(dft_wfn.molecule(), "ORBITAL",
                                              psi4.core.get_global_option("BASIS"),
                                              puream=dft_wfn.basisset().has_puream(),
                                              return_atomlist=True)
    sad_fitting_list = psi4.core.BasisSet.build(dft_wfn.molecule(), "DF_BASIS_SAD",
                                                psi4.core.get_option("SCF", "DF_BASIS_SAD"),
                                                puream=dft_wfn.basisset().has_puream(),
                                                return_atomlist=True)

    # Use Psi4 SADGuess object to build the SAD Guess
    SAD = psi4.core.SADGuess.build_SAD(dft_wfn.basisset(), sad_basis_list)
    SAD.set_atomic_fit_bases(sad_fitting_list)
    SAD.compute_guess()
    D_AO = SAD.Da()

    if run_fci:
        psi4.set_options({'ci_maxiter': 100})
        fci_e = psi4.energy('fci', return_wfn=False)

    # Initialize the potential object
    V_xc = dft_wfn.Da().clone()
    Vpotential = dft_wfn.V_potential()

    mints = psi4.core.MintsHelper(dft_wfn.basisset())
    I = np.asarray(mints.ao_eri())

    output_file = working_directory + "examples/results/H{}_R{}_{}_{}_nshots{}_layer{}_maxiter{}_resampling{}_slopeSPSA{}_slopeSCF{}_{}.dat".format(n_orbs,R,basis,opt_method,nshots,n_blocks,opt_maxiter,resampling,slope_SPSA,slope_SCF,simulation)

    print("*"*50)
    print("*" + " "*18 + "R = {:8.3f}".format(R) + " "*18 + "*")
    print("*"*50)
    with open(output_file,'w') as f: f.write('')

    Etot        = 0
    Delta_E     = 1e8
    dRMS        = 1e8
    Fock_list   = []
    DIIS_error  = []

    with open(output_file,'a') as f: f.write(str(circuits[0].decompose())+"\n")

    last_energies = []
    for SCF_ITER in range(1,SCF_maxiter+1):

        print("iteration ",SCF_ITER)

        # Compute the Coulomb potential
        J_coulomb = np.einsum('pqrs,rs->pq', I, D_AO.np)
        # Compute the XC potential with the new density matrix in the AO basis
        Vpotential.set_D([D_AO])
        Vpotential.compute_V([V_xc])

        # Compute the Fock matrix in the AO basis:
        F_AO = Hcore.np + 2*J_coulomb + V_xc.np

        # DIIS
        diis_e = np.einsum('ij,jk,kl->il', F_AO, D_AO.np, S_AO) - np.einsum('ij,jk,kl->il', S_AO, D_AO.np, F_AO)
        diis_e = S_sqrt_inv @ diis_e @ S_sqrt_inv
        Fock_list.append(F_AO)
        DIIS_error.append(diis_e)
        dRMS = np.mean(diis_e**2)**0.5
        param_values = param_values + np.random.rand(n_param)/10000.

        if SCF_ITER >= 2:
            # Limit size of DIIS vector
            diis_count = len(Fock_list)
            if diis_count > 6:
                # Remove oldest vector
                del Fock_list[0]
                del DIIS_error[0]
                diis_count -= 1

            # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
            B = np.empty((diis_count + 1, diis_count + 1))
            B[-1, :] = -1
            B[:, -1] = -1
            B[-1, -1] = 0
            for num1, e1 in enumerate(DIIS_error):
                for num2, e2 in enumerate(DIIS_error):
                    if num2 > num1: continue
                    val = np.einsum('ij,ij->', e1, e2)
                    B[num1, num2] = val
                    B[num2, num1] = val

            # normalize
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

            # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
            resid = np.zeros(diis_count + 1)
            resid[-1] = -1

            # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
            ci = np.linalg.solve(B, resid)

            # Calculate new fock matrix as linear
            # combination of previous fock matrices
            F_AO = np.zeros_like(F_AO)
            for num, c in enumerate(ci[:-1]):
                F_AO += c * Fock_list[num]

        # Build the Fock matrix in the OAO basis:
        F_OAO = S_sqrt_inv @ F_AO @ S_sqrt_inv

        # Map the non-interacting Hamiltonian (Fock matrix) in the OAO basis into Qubit Hamiltonian
        H_qubit, IJ_op_list, full_pauliop_list, full_pauli_list = QDFT.transformation_Hmatrix_Hqubit(F_OAO,n_qubits)
        solver  = NumPyEigensolver(k = 2**n_qubits)
        result  = solver.compute_eigenvalues(H_qubit)
        eigvals = result.eigenvalues
        eigvecs = result.eigenstates.to_matrix().T

        with open(output_file,'a') as f:
         f.write("varepsilon exact: {}".format(dft_wfn.epsilon_a().np[:]) + "\n")
         f.write("varepsilon from diag: {}".format(eigvals[:n_occ]) + "\n")
         f.write("weights: {}".format(weights) + "\n")

        if opt_method == "L-BFGS-B": opt_options = {'maxiter': opt_maxiter,'ftol':ftol,'gtol':gtol}
        if opt_method == "SLSQP": opt_options = {'maxiter': opt_maxiter,'ftol':ftol}
        if opt_method != "SPSA":
         if not basinhopping:
          f_min = scipy.optimize.minimize(QDFT.cost_function_energy,
                  x0      = (param_values),
                  args    = (circuits,
                             H_qubit,
                             weights,
                             simulation,
                             nshots,
                             backend,
                             False),
                  method  = opt_method,
                  options = opt_options)
         else:
          f_min = scipy.optimize.basinhopping(QDFT.cost_function_energy,
                  x0      = (param_values),
                  niter   = niter_hopping, # niter steps of basinhopping, and niter+1 optimizer iterations for each step.
                  minimizer_kwargs = {'method': opt_method,
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
          spsa = SPSA(maxiter=opt_maxiter,blocking=blocking,allowed_increase=allowed_increase,last_avg=slope_SPSA,resamplings=resampling)
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
          E_SA = result.fun # if last_avg is not 1, it returns the callable function with the last_avg param_values as input ! This seems generally good and better than taking the mean of the last_avg function calls.
          stddev = spsa.estimate_stddev(cost_function, initial_point=param_values)

        bounds = [circuits[state].bind_parameters(param_values) for state in range(n_occ)]
        if nshots is not False:
           energies = [sampled_expectation_value(bounds[state],H_qubit,backend,simulation,nshots=nshots) for state in range(n_occ)]
        else:
           energies = [np.real((StateFn(H_qubit, is_measurement=True) @ StateFn(bounds[state])).eval()) for state in range(n_occ)]

        with open(output_file,'a') as f: f.write("SA ENERGY: {}".format(E_SA) + "\n")

        if simulation == "noiseless": 
          bounds = [circuits[state].bind_parameters(param_values) for state in range(n_occ)]
          states = [np.array(Statevector(bounds[state])) for state in range(n_occ)]
          if nshots is not False:
            #states = [sampled_state(bounds[state],n_qubits,simulation,backend=backend,nshots=nshots) for state in range(n_occ)] # Grover... actually not needed.
            proba_states = [np.random.multinomial(nshots,(np.abs(states[state])**2)) / nshots for state in range(n_occ)]
          else: # state vector simulation
            proba_states = [np.abs(states[state])**2 for state in range(n_occ)]
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
            proba_state = np.random.multinomial(nshots,(np.abs(state)**2)) / nshots
            states.append(state)
            proba_states.append(proba_state)
        else:
          sys.exit("Simulation argument '{}' does not exist".format(simulation))

        # evaluate all the expectation values of Pauli strings to compute the density:
        bounds = [circuits[state].bind_parameters(param_values) for state in range(n_occ)]
        all_exp_values = np.zeros((n_occ,len(full_pauliop_list)))
        for state in range(n_occ):
          for pauli in range(len(full_pauliop_list)):
            if nshots is not False: 
              all_exp_values[state,pauli] = sampled_expectation_value(bounds[state],full_pauliop_list[pauli],backend,simulation,nshots=nshots)
            else: 
              all_exp_values[state,pauli] = np.real((StateFn(full_pauliop_list[pauli], is_measurement=True) @ StateFn(bounds[state])).eval())

        # Use the expectation values of Pauli strings to estimate the expectation value of |Phi_I><Phi_J| + h.c.
        product_coeff = np.zeros((n_occ,n_orbs*(n_orbs+1)//2))
        if len(IJ_op_list) != len(product_coeff[0]): sys.exit("dimension problem in the number of IJ_op elements")
        for occ in range(n_occ):
          for ij in range(len(IJ_op_list)):
              for pauli in IJ_op_list[ij]:
                # product coeff is obtained from < psi_k |( | Phi_I > < Phi_J | + h.c. )| psi_k > = 2 Re( c_Ik c_Jk )
                product_coeff[occ,ij] += pauli.coeffs[0] * all_exp_values[occ,full_pauli_list.index(pauli.to_pauli_op().primitive)] / 2.

        # Compute the density matrix
        D_OAO = np.zeros((n_orbs,n_orbs))
        for i in range(n_orbs):
         for j in range(i+1):
          for occ in range(n_occ):
            ij = i*(i+1)//2 + j
            D_OAO[i,j] += product_coeff[occ,ij]
          D_OAO[j,i] = D_OAO[i,j]
        D_AO.np[:] = S_sqrt_inv @ D_OAO @ S_sqrt_inv

        # Compute the Hxc energy with the new density:
        Hxcpot_energy = 2*np.einsum('pq,pq->', (J_coulomb + V_xc.np), D_AO.np) # factors 2 because D_AO is only alpha-D_AO.
        Vpotential.set_D([D_AO])
        Vpotential.compute_V([V_xc]) # otherwise it doesn't change the EHxc energy...
        EHxc = Vpotential.quadrature_values()["FUNCTIONAL"]
        # Compute the Hxc potential contribution (old potential * new density !)
        Etot_new = 2*np.sum(energies) + EHxc - Hxcpot_energy + E_nuc
        Delta_E  = abs(Etot_new - Etot)
        Etot     = Etot_new

        fidelities = np.array([abs(np.conj(states[i]).T @ C_OAO[:,i])**2 for i in range(n_occ)])
        if isinstance(fidelities,int): fidelities = [fidelities]
        if nshots is not False: 
           fidelities_proba = np.array([np.linalg.norm(proba_states[i] - proba_states_exact[i]) for i in range(n_occ)])
           if isinstance(fidelities_proba,int): fidelities_proba = [fidelities_proba]

        # Print results
        with open(output_file,'a') as f:
         f.write("*" * 10 + " ITERATION {:3d} ".format(SCF_ITER) + "*" * 10 + "\n")
         f.write("Energy (hartree) : {:16.8f}".format(Etot) + "\n")
         f.write("Occupied KS nrj  : {}".format(energies) + "\n")
         f.write("Fidelity wrt ED  : {}".format([abs(np.conj(states[i]).T @ eigvecs[:, i]) ** 2 for i in range(n_occ)]) + "\n")
         f.write("Fidelity wrt KS  : {}".format(fidelities) + "\n")
         if nshots is not False: f.write("DiffNorm proba   : {}".format(fidelities_proba) + "\n")
         f.write("Delta E iter     : {:16.8f}".format(Delta_E) + "\n")
         f.write("Delta E DFTexact : {:16.8f}".format(Etot - dft_e) + "\n")
         f.write("dRMS             : {:16.8f}\n".format(dRMS) + "\n")

        if slope_SCF is not None:
          last_energies.append(Etot)

          if len(last_energies) > slope_SCF:
              last_energies = last_energies[-slope_SCF:]
              pp = np.polyfit(range(slope_SCF), last_energies, 1)
              slope = pp[0]
              origin_intersection = pp[1]
              with open(output_file,'a') as f:
                 f.write("Slope SCF        : {:16.8f}".format(slope) + "\n")
                 f.write("Origin Inters.   : {:16.8f}\n".format(origin_intersection) + "\n")

              if abs(slope) < 5e-4:
                  Delta_E = 0.
                  normocc = 0.

        if ((Delta_E < E_conv) and (dRMS < D_conv)) or SCF_ITER == SCF_maxiter:

            if run_fci: rel_error_EVQE_EFCI = abs((Etot-fci_e)/fci_e)
            if run_fci: rel_error_EDFT_EFCI = abs((dft_e-fci_e)/fci_e)
            rel_error_EVQE_EDFT = abs((Etot-dft_e)/dft_e)
            rel_error_DVQE_DDFT = abs((np.linalg.norm(D_AO.np) - np.linalg.norm(D_AO_DFT.np))/np.linalg.norm(D_AO_DFT.np))

            with open(output_file,'a') as f:
             if SCF_ITER == SCF_maxiter: f.write("*"*10 + " FAILURE " + "*"*10 + "\n")
             if (Delta_E < E_conv) and (dRMS < D_conv): f.write("*"*10 + " SUCCESS " + "*"*10 + "\n")
             f.write("Iteration        : {:16d}".format(SCF_ITER) + "\n")
             f.write("States fidelity  : {}".format(fidelities) + "\n")
             if nshots is not False: f.write("DiffNorm proba   : {}".format(fidelities_proba) + "\n")
             f.write("Rel Err DVQE/DDFT: {:16.8f}".format(rel_error_DVQE_DDFT) + "\n")
             f.write("Rel Err EVQE/EDFT: {:16.8f}".format(rel_error_EVQE_EDFT) + "\n")
             if run_fci:
              f.write("Rel Err EVQE/EFCI: {:16.8f}".format(rel_error_EVQE_EFCI) + "\n")
              f.write("Rel Err EDFT/EFCI: {:16.8f}".format(rel_error_EDFT_EFCI) + "\n")
              f.write("FCI energy       : {:16.8f}".format(fci_e) + "\n")
             f.write("DFT energy exact : {:16.8f}".format(dft_e) + "\n")
             f.write("DFT energy SAVQE : {:16.8f}".format(Etot.real) + "\n")
             if opt_method == "SPSA": f.write("stddev SPSA      : {:16.8f}".format(stddev) + "\n")
            break

