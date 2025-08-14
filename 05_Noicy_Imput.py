#https://pennylane.ai/codebook/noisy-quantum-theory/all-mixed-up

#1
def build_density_matrix(state_1, state_2, p_1, p_2):

    projector_1 = np.outer(state_1, np.conj(state_1))
    projector_2 = np.outer(state_2, np.conj(state_2))   
    
    # The density matrix is the weighted sum of the projectors,
    # where the weights are the probabilities.
    density_matrix = p_1 * projector_1 + p_2 * projector_2
    
    return density_matrix

print("state_1 = |+y>, state_2 = |+x>, p_1 = 0.5, p_2 = 0.5")
print("density_matrix:")
print(build_density_matrix([1,1j]/np.sqrt(2), [1,1]/np.sqrt(2), 0.5, 0.5))   


#2

def is_hermitian(matrix):

    
    return np.allclose(matrix, np.conj(matrix).T)

matrix_1 = np.array([[1,1j],[-1j,1]])
matrix_2 = np.array([[1,2],[3,4]])

print("Is matrix [[1,1j],[-1j,1]] Hermitian?")
print(is_hermitian(matrix_1))
print("Is matrix [[1,2],[3,4]] Hermitian?")
print(is_hermitian(matrix_2))




#3



def has_trace_one(matrix):

    
    return np.isclose(np.trace(matrix), 1)

matrix_1 = [[1/2,1j],[-1j,1/2]]
matrix_2 = [[1,2],[3,4]]
    
print("Does [[1/2,1j],[-1j,1/2]] have unit trace?")
print(has_trace_one(matrix_1))
print("Does [[1,2],[3,4]] have unit trace?")
print(has_trace_one(matrix_2))





#4

def is_semi_positive(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return bool(np.all(eigenvalues >= -1e-9))
    
matrix_1 = [[3/4,1/4],[1/4,1/4]]
matrix_2 = [[0,1/4],[1/4,1/4]]

print("Is matrix [[3/4,1/4],[1/4,1/4]] positive semidefinite?")
print(is_semi_positive(matrix_1))
print("Is matrix [[0,1/4],[1/4,1/4]] positive semidefinite?")
print(is_semi_positive(matrix_2))



#5

def is_density_matrix(matrix):    
    return is_hermitian(matrix) and has_trace_one(matrix) and is_semi_positive(matrix) 

matrix_1 = np.array([[3/4,0.25j],[-0.25j,1/4]])
matrix_2 = np.array([[0,1/4],[1/4,1/4]])
    
print("Is [[3/4,0.25j],[-0.25j,1/4]] a density matrix?")
print(is_density_matrix(matrix_1))
print("Is matrix [[0,1/4],[1/4,1/4]] a density matrix?")
print(is_density_matrix(matrix_2))



#6


def purity(density_matrix):  
    return np.trace(density_matrix @ density_matrix).real

matrix_1 = np.array([[1/2,1/2],[1/2,1/2]])
matrix_2 = np.array([[3/4,1/4],[1/4,1/4]])

print("The purity of [[1/2,1/2],[1/2,1/2]] is {}".format(purity(matrix_1)))
print("The purity of [[3/4,1/4],[1/4,1/4]] is {}".format(purity(matrix_2)))



#https://pennylane.ai/codebook/noisy-quantum-theory/operations-with-mixed-states


#1


def apply_gate_mixed(rho,U):

    # Apply the evolution formula rho_new = U * rho * U_dagger
    # using matrix multiplication (@)
    return U @ rho @ (U.conj().T)

U = qml.matrix(qml.RX(np.pi/3,0))
rho = np.array([[3/4,1/4],[1/4,1/4]])

print("A pi/3 RX rotation applied on [[3/4,1/4],[1/4,1/4]] gives:")
print(apply_gate_mixed(rho,U).round(2))



#2


dev = qml.device("default.mixed", wires=1)

@qml.qnode(dev)
def apply_gate_circuit(rho,U):

    qml.QubitDensityMatrix(rho, wires=0)
    qml.QubitUnitary(U, wires=0)
    return qml.state()
  
U = qml.matrix(qml.RX(np.pi/3,0))
rho = np.array([[3/4,1/4],[1/4,1/4]])

print("A pi/3 RX rotation applied on [[3/4,1/4],[1/4,1/4]] gives:")
print(apply_gate_circuit(rho,U).round(2))


#3

def eigenprojectors(B):
    
    eigenvectors = np.transpose(np.linalg.eig(B)[1])

    eigen_projectors = np.array([np.outer(vector,np.conj(vector)) for vector in eigenvectors])

    return eigen_projectors

#4


def outcome_probs(rho, B):
    eigen_projectors = eigenprojectors(B)
    return np.array([np.trace(rho @ proj).real for proj in eigen_projectors])
    

rho = np.array([[3/4,0],[0,1/4]])
B = qml.matrix(qml.PauliY(0))

print("State: [[3/4,0],[0,1/4]], Observable: {}".format(B))
print("Measurement probabilities {}".format(outcome_probs(rho,B).round(2)))




#5


dev = qml.device('default.mixed', wires = 1)

@qml.qnode(dev)
def mixed_probs_circuit(rho, obs):
    # Prepare the density matrix rho on the wire(s) specified by the observable.
    qml.QubitDensityMatrix(rho, wires=obs.wires)
    return qml.probs(op=obs)

rho = np.array([[3/4,0],[0,1/4]])
B = qml.PauliY(0)

print("State: [[3/4,0],[0,1/4]], Observable: {}".format(qml.matrix(B)))
print("Measurement probabilities {}".format(mixed_probs_circuit(rho,B))) 



#6


def mixed_expval(rho, B):
    return np.trace(rho@B)


rho = np.array([[3/4,0],[0,1/4]])
B = qml.matrix(qml.PauliZ(0))

print("State: {}".format(rho))
print("Observable: {}".format(B))
print("Expectation value: {}".format(mixed_expval(rho,B)))



#7


dev = qml.device('default.mixed', wires = 1)

@qml.qnode(dev)
def expval_circuit(rho,obs):
    qml.QubitDensityMatrix(rho, wires=0)
    return qml.expval(obs)

rho = np.array([[3/4,0],[0,1/4]])
B = qml.PauliZ(0)

print("State: {}".format(rho))
print("Observable: {}".format(qml.matrix(B)))
print("Expectation value: {}".format(expval_circuit(rho,B)))





#https://pennylane.ai/codebook/noisy-quantum-theory/trace-it-and-forget-it



#1


def composite_density_matrix(rho, sigma):
    return np.kron(rho, sigma)


#2

def create_entangled(alpha):
    # Write your circuit here
    qml.RY(alpha, wires=0)
    qml.CNOT(wires=[0, 1])

#3


dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def reduced_entangled(alpha):
    # Prepare the state using create_entangled
    create_entangled(alpha)
    return qml.density_matrix(wires=0)

alpha = np.pi/3

print("For alpha = pi/3, the reduced density matrix is {}".format(reduced_entangled(alpha)))




#4


dev = qml.device('default.mixed', wires = 2)

parity_even = 0.5*qml.PauliZ(wires=0) @ qml.PauliZ(wires=1)+ 0.5*qml.Identity(0) @ qml.Identity(1)
parity_odd = - 0.5*qml.PauliZ(wires=0) @ qml.PauliZ(wires=1)+ 0.5*qml.Identity(0) @ qml.Identity(1)

max_mixed = np.eye(4)/4
psi_plus = qml.math.dm_from_state_vector(np.array([1,0,0,1])/np.sqrt(2))

@qml.qnode(dev)
def parity_check_circuit(state,parity_operator):
    qml.QubitDensityMatrix(state, wires=[0, 1])
    return qml.expval(parity_operator)
   


print("Maximal mixed state expected values")
print(f"Odd Parity: {parity_check_circuit(max_mixed,parity_odd)}")
print(f"Even Parity: {parity_check_circuit(max_mixed,parity_even)}")

print("Maximal entangled state expected values")
print(f"Odd Parity: {parity_check_circuit(psi_plus,parity_odd)}")
print(f"Even Parity: {parity_check_circuit(psi_plus,parity_even)}")
