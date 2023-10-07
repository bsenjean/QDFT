from qiskit.opflow import I, X, Y
from qiskit.opflow.primitive_ops import PauliOp

def transformation_Hmatrix_Hqubit(Hmatrix,nqubits):
    """
    This function transforms the Hamiltonian matrix into the qubit Hamiltonian.
    Each element of the Hamiltonian is defined in the basis of bitstring configurations,
    i.e. | 0000 ... 0 >, | 0000 ... 1 >, etc. Each value "1" corresponds to a qubit excitation
    that is given by the operation S^+.
    """

    H_qubit = 0.
    IJ_op_list = [] # list of operators for ( |phi_I><phi_J| + h.c. )
    full_pauliop_list = [] # list of pauli strings operator to measure the Hamiltonian and c_i c_j product for the density.
    full_pauli_list = [] # list of pauli strings to measure the Hamiltonian and c_i c_j product for the density.

    for i in range(len(Hmatrix[0])):
     # | I >                          
     for j in range(len(Hmatrix[0])):
       # < J |
       IJ_op = I^nqubits

       # First convert the integers into bitstrings:
       bitstring_i = bin(i)[2:].zfill(nqubits)
       bitstring_j = bin(j)[2:].zfill(nqubits)

       for qubit in range(nqubits):
         if int(list(bitstring_i)[qubit]) == 0 and int(list(bitstring_j)[qubit]) == 0:
            IJ_op = IJ_op @ ((I^qubit) ^ (((X + 1j*Y)@(X - 1j*Y))/4.) ^ (I^(nqubits - qubit - 1)))
         elif int(list(bitstring_i)[qubit]) == 0 and int(list(bitstring_j)[qubit]) == 1:
            IJ_op = IJ_op @ ((I^qubit) ^ ( (X + 1j*Y)            /2.) ^ (I^(nqubits - qubit - 1)))
         elif int(list(bitstring_i)[qubit]) == 1 and int(list(bitstring_j)[qubit]) == 0:
            IJ_op = IJ_op @ ((I^qubit) ^ (            (X - 1j*Y) /2.) ^ (I^(nqubits - qubit - 1)))
         elif int(list(bitstring_i)[qubit]) == 1 and int(list(bitstring_j)[qubit]) == 1:
            IJ_op = IJ_op @ ((I^qubit) ^ (((X - 1j*Y)@(X + 1j*Y))/4.) ^ (I^(nqubits - qubit - 1)))

       H_qubit += float(Hmatrix[i,j]) * IJ_op.reduce()

       IJ_op_adjoint = IJ_op.adjoint()
       IJ_op = IJ_op + IJ_op_adjoint

       IJ_op = IJ_op.reduce()
       if j < i+1:
         IJ_op_list.append(IJ_op)
         for num_pauli in range(len(IJ_op.to_pauli_op())):
           pauli_string = IJ_op.to_pauli_op()[num_pauli].primitive
           if pauli_string not in full_pauli_list:
              full_pauliop_list.append(PauliOp(pauli_string))
              full_pauli_list.append(pauli_string)

    return H_qubit.reduce(), IJ_op_list,full_pauliop_list, full_pauli_list

def sz_operator(n_qubits):
    s_z = 0

    for i in range(n_qubits//2):
      s_z += FermionicOp(("N_{}".format(2*i),0.5),register_length=n_qubits)
      s_z -= FermionicOp(("N_{}".format(2*i+1),0.5),register_length=n_qubits)

    return s_z

def s2_operator(n_qubits):
    '''
    S2 = S- S+ + Sz(Sz+1)
    I use the usual sorting as in OpenFermion, i.e. 1up 1down, 2up 2down, etc...
    '''
    s2_op = 0
    s_moins = 0
    s_plus = 0
    s_z = sz_operator(n_qubits)

    for i in range(n_qubits//2):
      s_moins += FermionicOp(("+_{} -_{}".format(2*i+1,2*i),1),register_length=n_qubits)
      s_plus += FermionicOp(("+_{} -_{}".format(2*i,2*i+1),1),register_length=n_qubits)

    s2_op = s_moins @ s_plus + s_z @ s_z + s_z
    return s2_op
