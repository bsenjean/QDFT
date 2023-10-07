import os,sys
import numpy as np
import math
sys.path.insert(1, os.path.abspath('/usr/local/psi4/lib/'))
import psi4

#psi4.set_memory('30 GB')

working_directory = os.getenv('QDFT_DIR') + "/examples/"

#===========================================================#
#=============== Initialization by the user ================#
#===========================================================#

functional  = "SVWN"
basis       = "sto-3g"
E_conv      = 1e-9
D_conv      = 1e-6
SCF_maxiter = 100
n_hydrogens = 4
n_elec      = n_hydrogens
n_occ       = n_elec//2
#interdist_list = [0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
interdist_list = [4.0]

#===========================================================#
#=============== End of the initialization =================#
#===========================================================#

#========================================================================#
#============= START THE ALGORITHM FOR DIFFERENT U VALUES ===============#
#========================================================================#
for R in interdist_list:

    psi4.core.clean()
    # Define the geometry
    psi4.core.set_output_file(working_directory + "results/H{}_R{}_{}_{}_Psi4.dat".format(n_hydrogens,R,basis,functional),True)
    string_geo  = "0 1\n"
    for d in range(n_hydrogens//2):
      string_geo += "H 0. 0. {}\n".format(- (R/2. + d*R))
      string_geo += "H 0. 0. {}\n".format(+ (R/2. + d*R))
    string_geo += "symmetry c1\n"
    string_geo += "nocom\n"
    string_geo += "noreorient\n"

    psi4.geometry(string_geo)

    psi4.set_options({'basis': basis,'save_jk':True, 'debug':1, 'print':5, 'scf_type':'pk'})
    #psi4.set_options({'basis': basis,'scf_type':'pk'})
    dft_e, dft_wfn = psi4.energy(functional, return_wfn=True)

    # Hcore matrix:
    Hcore = dft_wfn.H().clone()
    # Nuclear energy:
    E_nuc = dft_wfn.get_energies('Nuclear')
    # Overlap matrix in the AO basis:
    S_AO = dft_wfn.S().np
    # Compute the inverse square root of the overlap matrix S
    S_eigval, S_eigvec = np.linalg.eigh(S_AO)
    S_sqrt_inv = S_eigvec @ np.diag((S_eigval)**(-1./2.)) @ S_eigvec.T
    C_transformation = np.linalg.inv(S_sqrt_inv)

    # Construct the SAD Guess for the initial density D_AO
    psi4.core.prepare_options_for_module("SCF")
    sad_basis_list = psi4.core.BasisSet.build(dft_wfn.molecule(), "ORBITAL",
        psi4.core.get_global_option("BASIS"), puream=dft_wfn.basisset().has_puream(),
                                         return_atomlist=True)
    sad_fitting_list = psi4.core.BasisSet.build(dft_wfn.molecule(), "DF_BASIS_SAD",
        psi4.core.get_option("SCF", "DF_BASIS_SAD"), puream=dft_wfn.basisset().has_puream(),
                                           return_atomlist=True)

    # Use Psi4 SADGuess object to build the SAD Guess
    SAD = psi4.core.SADGuess.build_SAD(dft_wfn.basisset(), sad_basis_list)
    SAD.set_atomic_fit_bases(sad_fitting_list)
    SAD.compute_guess();
    D_AO = SAD.Da().clone()

    # Initialize the potential object
    V_xc = dft_wfn.Da().clone()
    Vpotential = dft_wfn.V_potential()

    mints = psi4.core.MintsHelper(dft_wfn.basisset())
    I = np.asarray(mints.ao_eri())
    AO_potential = np.asarray(mints.ao_potential())
    AO_kinetic = np.asarray(mints.ao_kinetic())

    Delta_E     = 1e8
    dRMS        = 1e8
    Fock_list = []
    DIIS_error = []

    # Compute the energy of the SAD guess:
    # Coulomb potential
    J_coulomb = np.einsum('pqrs,rs->pq', I, D_AO.np)
    # XC potential
    Vpotential.set_D([D_AO])
    Vpotential.compute_V([V_xc])
    # Compute the Hxc energy:
    EH = 2*np.einsum('pq,pq->', J_coulomb, D_AO.np)
    Exc = Vpotential.quadrature_values()["FUNCTIONAL"]
    EHxc = EH + Exc
    # Compute the kinetic energy
    Ts = 2*np.einsum('pq,pq->', AO_kinetic, D_AO.np)
    # Compute the energy contributions from the potentials:
    Hxcpot_energy = 2*np.einsum('pq,pq->', (2*J_coulomb + V_xc.np), D_AO.np)
    Extpot_energy = 2*np.einsum('pq,pq->', AO_potential, D_AO.np)
    Etot = Ts + EHxc + Extpot_energy + E_nuc
    print("SAD guess energy: {:24.16f}".format(Etot))

    # Get the first density matrix after SAD guess:
    F_AO = Hcore.np + 2*J_coulomb + V_xc.np
    F_OAO = S_sqrt_inv @ F_AO @ S_sqrt_inv
    eigvals,eigvecs = np.linalg.eigh(F_OAO)
    # Transform back to the AO basis:
    C_MO = S_sqrt_inv @ eigvecs
    C_occ = S_sqrt_inv @ eigvecs[:,:n_occ] # AO --> OAO = C_transformation = S^{1/2}, OAO --> AO = S^{-1/2}
    # Compute the alpha-density matrix
    D_AO.np[:] = np.einsum('pi,qi->pq', C_occ, C_occ, optimize=True)

    for SCF_ITER in range(1, SCF_maxiter+1):

        ##########################################################################
        # FIRST WAY TO COMPUTE THE ENERGY, WITHOUT DIAGONALIZING THE FOCK MATRIX #
        ##########################################################################

        # Update the potential from the new density
        J_coulomb = np.einsum('pqrs,rs->pq', I, D_AO.np)
        Vpotential.set_D([D_AO])
        Vpotential.compute_V([V_xc]) # Also required to compute the Exc energy

        # Build the Fock matrix (for DIIS)
        F_AO = Hcore.np + 2*J_coulomb + V_xc.np

        # DIIS
        diis_e = np.einsum('ij,jk,kl->il', F_AO, D_AO.np, S_AO) - np.einsum('ij,jk,kl->il', S_AO, D_AO.np, F_AO)
        diis_e = S_sqrt_inv @ diis_e @ S_sqrt_inv
        Fock_list.append(F_AO)
        DIIS_error.append(diis_e)
        dRMS = np.mean(diis_e**2)**0.5

        # Compute the energy contributions from the potentials, with the density of the same iteration...
        Hxcpot_energy = 2*np.einsum('pq,pq->', (2*J_coulomb + V_xc.np), D_AO.np)
        Extpot_energy = 2*np.einsum('pq,pq->', AO_potential, D_AO.np)

        # Compute the Hartree energy
        EH = 2*np.einsum('pq,pq->', J_coulomb, D_AO.np)
        # Compute the XC energy
        Exc = Vpotential.quadrature_values()["FUNCTIONAL"]
        EHxc = EH + Exc

        # Compute the kinetic energy
        Ts = 2*np.einsum('pq,pq->', AO_kinetic, D_AO.np)

        Etot_new = Ts + EHxc + Extpot_energy + E_nuc

        ######################################################################
        # SECOND WAY TO COMPUTE THE ENERGY, BY DIAGONALIZING THE FOCK MATRIX #
        ######################################################################

        F_OAO = S_sqrt_inv @ F_AO @ S_sqrt_inv
        eigvals,eigvecs = np.linalg.eigh(F_OAO)
        # Transform back to the AO basis:
        C_MO = S_sqrt_inv @ eigvecs
        C_occ = S_sqrt_inv @ eigvecs[:,:n_occ] # AO --> OAO = C_transformation = S^{1/2}, OAO --> AO = S^{-1/2}
        # Compute the alpha-density matrix
        D_AO.np[:] = np.einsum('pi,qi->pq', C_occ, C_occ, optimize=True)
        # Compute the energy contributions from the old potential with the new density
        Hxcpot_energy = 2*np.einsum('pq,pq->', (2*J_coulomb + V_xc.np), D_AO.np)
        Extpot_energy = 2*np.einsum('pq,pq->', AO_potential, D_AO.np)
        # Compute the Hxc energy:
        J_coulomb = np.einsum('pqrs,rs->pq', I, D_AO.np)
        EH = 2*np.einsum('pq,pq->', J_coulomb, D_AO.np)
        Vpotential.set_D([D_AO])
        Vpotential.compute_V([V_xc])
        Exc = Vpotential.quadrature_values()["FUNCTIONAL"]
        EHxc = EH + Exc
        Etot_new2 = 2*np.sum(eigvals[:n_occ]) + EHxc - Hxcpot_energy + E_nuc

        # Print results
        #print("compare (Ts):",2*np.einsum('pq,pq->', AO_kinetic, D_AO.np),2*np.sum(eigvals[:n_occ])-Hxcpot_energy-Extpot_energy)
        print("Energy (hartree) : {:24.16f} {:24.16f}".format(Etot_new,Etot_new2))

        Delta_E  = abs(Etot_new - Etot)
        Etot     = Etot_new

        if (Delta_E < E_conv) and (dRMS < D_conv):
            print("*"*10 + " SUCCESS " + "*"*10)
            print("R = ",R)
            print("Iteration        : {:16d}".format(SCF_ITER))
            print("DFT energy       : {:16.8f}".format(Etot))
            print("DFT energy Psi4  : {:16.8f}".format(dft_e))
            break

        if SCF_ITER == SCF_maxiter:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")

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
        eigvals,eigvecs = np.linalg.eigh(F_OAO)
        #print("varepsilon exacts    :",dft_wfn.epsilon_a().np[:])
        #print("varepsilon iteration :",eigvals)

        # Transform back to the AO basis:
        C_MO = S_sqrt_inv @ eigvecs
        C_occ = S_sqrt_inv @ eigvecs[:,:n_occ] # AO --> OAO = C_transformation = S^{1/2}, OAO --> AO = S^{-1/2}

        # Compute the alpha-density matrix
        D_AO.np[:] = np.einsum('pi,qi->pq', C_occ, C_occ, optimize=True)
