from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit_nature.second_q.operators import FermionicOp
import numpy as np

'''decompose operator to pauli'''
# Example numpy ndarray representing an operator
operator_matrix = np.array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])

# Define Pauli matrices and the identity matrix
II = Pauli('II')
IX = Pauli('IX')
IY = Pauli('IY')
IZ = Pauli('IZ')
XI = Pauli('XI')
XX = Pauli('XX')
XY = Pauli('XY')
XZ = Pauli('XZ')
YI = Pauli('YI')
YX = Pauli('YX')
YY = Pauli('YY')
YZ = Pauli('YZ')
ZI = Pauli('ZI')
ZX = Pauli('ZX')
ZY = Pauli('ZY')
ZZ = Pauli('ZZ')

# Map Pauli matrices to their corresponding ndarray representation
pauli_to_matrix = {
    'II': np.kron(np.eye(2), np.eye(2)),
    'IX': np.kron(np.eye(2), np.array([[0, 1], [1, 0]])),
    'IY': np.kron(np.eye(2), np.array([[0, -1j], [1j, 0]])),
    'IZ': np.kron(np.eye(2), np.array([[1, 0], [0, -1]])),
    'XI': np.kron(np.array([[0, 1], [1, 0]]), np.eye(2)),
    'XX': np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])),
    'XY': np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])),
    'XZ': np.kron(np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]])),
    'YI': np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(2)),
    'YX': np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, 1], [1, 0]])),
    'YY': np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]])),
    'YZ': np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])),
    'ZI': np.kron(np.array([[1, 0], [0, -1]]), np.eye(2)),
    'ZX': np.kron(np.array([[1, 0], [0, -1]]), np.array([[0, 1], [1, 0]])),
    'ZY': np.kron(np.array([[1, 0], [0, -1]]), np.array([[0, -1j], [1j, 0]])),
    'ZZ': np.kron(np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]])),
}


# Function to decompose operator into Pauli strings
def decompose_operator_to_pauli_list(operator):

    pauli_list = {}

    # Get the dimension of the operator
    dim = operator.dim[0]

    # Iterate over all possible Pauli matrices
    for pauli_str, pauli_matrix in pauli_to_matrix.items():
        # Apply the Pauli matrix to the operator and compute the expectation value
        coeff = np.trace(np.dot(operator.data, pauli_matrix)) / dim

        # Check if coefficient is non-zero
        if np.abs(coeff) > 1e-10:  # Adjust threshold as needed
            if pauli_str in pauli_list:
                pauli_list[pauli_str] += coeff
            else:
                pauli_list[pauli_str] = coeff

    pauli_list = [(Pauli(pauli_str), coeff) for pauli_str, coeff in pauli_list.items()]

    return pauli_list


P = [Pauli('X'), Pauli('Y'), Pauli('Z'), Pauli('I')]
X, Y, Z, I = Operator(P[0]).data, Operator(P[1]).data, Operator(P[2]).data, Operator(P[3]).data


def list_Pauli_to_SparsePauliOp(list_pauli: list):

    pauli_list: list[Pauli] = []
    pauli_coeff: list[float] = []
    for lines in list_pauli:
        pauli_list.append(lines[0])
        pauli_coeff.append(lines[1])
    return SparsePauliOp(pauli_list, coeffs=pauli_coeff)


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
        for j in range(len(Hmatrix[0])):

            # < J |
            # i = | I >
            # j = < J |
            IJ_op = np.eye(2 ** nqubits)

            # First convert the integers into bitstrings:
            bitstring_i = bin(i)[2:].zfill(nqubits)
            bitstring_j = bin(j)[2:].zfill(nqubits)

            for qubit in range(nqubits):
                if int(bitstring_i[qubit]) == 0 and int(bitstring_j[qubit]) == 0:
                       op_1 = (((X + 1j*Y)@(X - 1j*Y))/4.)
                       product_op = Operator(op_1)
                       # Construct the full tensor product
                       left_identity = Operator(np.eye(2 ** qubit))
                       right_identity = Operator(np.eye(2 ** (nqubits - qubit - 1)))
                       full_op = left_identity.tensor(product_op).tensor(right_identity)
                       # Perform the matrix multiplication
                       IJ_op = Operator(IJ_op) @ full_op
                elif int(bitstring_i[qubit]) == 0 and int(bitstring_j[qubit]) == 1:
                       op_1 = (X + 1j*Y)/2.
                       product_op = Operator(op_1)
                       # Construct the full tensor product
                       left_identity = Operator(np.eye(2 ** qubit))
                       right_identity = Operator(np.eye(2 ** (nqubits - qubit - 1)))
                       full_op = left_identity.tensor(product_op).tensor(right_identity)
                       # Perform the matrix multiplication
                       IJ_op = Operator(IJ_op) @ full_op
                elif int(bitstring_i[qubit]) == 1 and int(bitstring_j[qubit]) == 0:
                       op_1 = (X - 1j*Y)/2.
                       product_op = Operator(op_1)
                       # Construct the full tensor product
                       left_identity = Operator(np.eye(2 ** qubit))
                       right_identity = Operator(np.eye(2 ** (nqubits - qubit - 1)))
                       full_op = left_identity.tensor(product_op).tensor(right_identity)
                       # Perform the matrix multiplication
                       IJ_op = Operator(IJ_op) @ full_op
                elif int(bitstring_i[qubit]) == 1 and int(bitstring_j[qubit]) == 1:
                       op_1 = (((X - 1j * Y) @ (X + 1j * Y)) / 4.)
                       product_op = Operator(op_1)
                       # Construct the full tensor product
                       left_identity = Operator(np.eye(2 ** qubit))
                       right_identity = Operator(np.eye(2 ** (nqubits - qubit - 1)))
                       full_op = left_identity.tensor(product_op).tensor(right_identity)
                       # Perform the matrix multiplication
                       IJ_op = Operator(IJ_op) @ full_op

                # Add the element of the Hamiltonian matrix to H_qubit
            H_qubit += Hmatrix[i, j] * IJ_op
            # Calculate the adjoint of IJ_op and add it
            IJ_op_adjoint = IJ_op.transpose()

            IJ_op = IJ_op + IJ_op_adjoint
            IJ_op = decompose_operator_to_pauli_list(IJ_op)
            if j < i+1:
                IJ_op_list.append(IJ_op)
                for num_pauli in range(len(IJ_op)):
                    pauli_string = IJ_op[num_pauli]
                    if pauli_string[0] not in full_pauli_list:
                        full_pauliop_list.append(SparsePauliOp(pauli_string[0]))
                        full_pauli_list.append(pauli_string[0])
    H_qubit = decompose_operator_to_pauli_list(H_qubit)
    H_qubit = list_Pauli_to_SparsePauliOp(H_qubit)
    # IJ_op_list = list_Pauli_to_SparsePauliOp(IJ_op_list)
    IJ_op_list_sp = []
    for list_values in IJ_op_list:
        IJ_op_list_sp.append(list_Pauli_to_SparsePauliOp(list_values))

    IJ_op_list = IJ_op_list_sp

    return H_qubit, IJ_op_list, full_pauliop_list, full_pauli_list


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
