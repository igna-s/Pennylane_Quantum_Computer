#https://pennylane.ai/codebook/hamiltonian-time-evolution/simulating-nature


#1


input = [1, 1, 1,1,1,0,0,1] # MODIFY EXAMPLE
output = secret_box(input)
print("The result of applying the secret box to ", input, "is ", output)
# We will secretly apply the function and return the result!

def deterministic_box(bits):

    if not bits:
        return []
    
    # The rule is a left circular shift
    return bits[1:] + bits[:1]

#2


input = 0 # MODIFY EXAMPLE
output = secret_box(input)
trials = 100  # INCREASE TRIALS TO IMPROVE APPROXIMATION
print("On input", input, "the approximate probability distribution is", output)
# We will secretly apply the function and return the result!

def random_box(bit):
    return np.random.choice(2)




#3



dev = qml.device("default.qubit", wires=1)

input = 0 # MODIFY EXAMPLE
reps = 2
output = secret_box(input, reps)
print("The probability distribution after applying the secret box to ", input)
print("a total of ", reps, "time(s) is ", output)
# We will secretly apply the function and return the result!

@qml.qnode(dev)
def quantum_box(bit, reps):
    if bit == 1:
        qml.PauliX(wires=0)
    for _ in range(reps):
        qml.Hadamard(wires=0)
      
    return qml.probs(wires=0)





#https://pennylane.ai/codebook/hamiltonian-time-evolution/unitaries


#1

def unitary_check(operator):

    adjoint = np.transpose(np.conjugate(operator))
    dim = operator.shape[0]
    return np.allclose(adjoint @ operator, np.identity(dim))



#2

n_bits = 1
dev = qml.device("default.qubit", wires=n_bits)

@qml.qnode(dev)
def unitary_circuit(operator):


    # Condición 1
    correct_size = (operator.shape == (2, 2))

    # Condición 2
    is_unitary = unitary_check(operator)

    # If both are true, we apply the operator.
    if correct_size and is_unitary:
        qml.QubitUnitary(operator, wires=0)


    return qml.state()




#https://pennylane.ai/codebook/hamiltonian-time-evolution/hamiltonians

#1

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def evolve_rotation(B, time):

    e = 1.6e-19
    m_e = 9.1e-31
    alpha = B*e/(2*m_e)
    qml.RZ(-2*alpha*time, wires=0)
    return qml.state()



#2


dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def evolve_magnetic(B, time):

    e = 1.6e-19
    m_e = 9.1e-31
    alpha = B*e/(2*m_e)
    qml.evolve(qml.PauliZ(0), -alpha*time)
    return qml.state()

# Now let's compare the solution of this method with the previous one!

B, t = 0.1, 0.6
if np.allclose(evolve_rotation(B, t), evolve_magnetic(B, t)):
    print("The two circuits give the same answer!")




#3



dev = qml.device("default.qubit", wires=1)

def evolve_plus_state(B, time):

    e = 1.6e-19
    m_e = 9.1e-31
    alpha = B*e/(2*m_e)
    qml.Hadamard(wires=0)
    qml.RZ(-2*alpha*time, wires=0)

@qml.qnode(dev)
def mag_z_plus_X(B, time):

    evolve_plus_state(B, time)
    return qml.expval(qml.PauliX(0))

@qml.qnode(dev)
def mag_z_plus_Y(B, time):

    evolve_plus_state(B, time)
    return qml.expval(qml.PauliY(0))

##################
# HIT SUBMIT FOR #
# PLOTTING MAGIC #
##################



#https://pennylane.ai/codebook/hamiltonian-time-evolution/energy-in-quantum-systems

#1

n_bits = 2
dev = qml.device("default.qubit", wires=n_bits)

@qml.qnode(dev)
def zz_circuit(alpha, time, init):

    qml.BasisState(init, wires=range(n_bits))
    
    # Implement the circuit CNOT-Rz-CNOT
    qml.CNOT(wires=[0, 1])
    qml.RZ(2 * alpha * time, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=range(n_bits))




#2


n_bits = 2
dev = qml.device("default.qubit", wires=n_bits)

@qml.qnode(dev)
def ising_circuit(alpha, time, init):

    qml.BasisState(init, wires=range(n_bits))

    qml.IsingZZ(2 * alpha * time, wires=[0, 1])
    return qml.state()


#3


n_bits = 2
dev = qml.device("default.qubit", wires=n_bits)

@qml.qnode(dev)
def ZZ_evolve(alpha, time, init):

    qml.BasisState(init, wires=range(n_bits))
    qml.evolve(alpha*qml.PauliZ(0)@qml.PauliZ(1), coeff = time)
    return qml.state()



#4


n_bits = 5
dev = qml.device("default.qubit", wires=n_bits)
    
coeffs = [1, 1, 1, 1]
obs = [
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliZ(1) @ qml.PauliZ(2),
    qml.PauliZ(1) @ qml.PauliZ(3),
    qml.PauliZ(3) @ qml.PauliZ(4)
]

H = qml.dot(coeffs, obs)

@qml.qnode(dev)
def energy(init):

    qml.BasisState(init, wires=range(n_bits))
    return qml.expval(H)





#5

my_guess1 = np.array([0,1,0,0,1]) # MODIFY THIS
my_guess2 = np.array([1,0,1,1,0]) # MODIFY THIS

print("The expected energy for", my_guess1, "is", energy(my_guess1), ".")
print("The expected energy for", my_guess2, "is", energy(my_guess2), ".")



#https://pennylane.ai/codebook/hamiltonian-time-evolution/approximating-exponentials


#1


n_bits=2
dev = qml.device("default.qubit", wires=range(n_bits))

@qml.qnode(dev)
def two_distant_spins(B, time):

    e = 1.6e-19
    m_e = 9.1e-31
    alpha = B*e/(2*m_e)  
    qml.RZ(-2 * alpha * time, wires=0)
    qml.RZ(-2 * alpha * time, wires=1)
    return qml.state()



#2

n_bits=2
dev = qml.device("default.qubit", wires=range(n_bits))

@qml.qnode(dev)
def two_close_spins_X(alpha, beta, time, n):


    for _ in range(n):
        qml.IsingXX(2*beta*time/n, wires=[0,1])
        qml.RZ(2*alpha*time/n, wires=[0])
        qml.RZ(2*alpha*time/n, wires=[1])
    return qml.state()


#3



n_bits=2
dev = qml.device("default.qubit", wires=range(n_bits))

def ham_close_spins(alpha, J):

    # Define the list of coefficients
    coeffs = [-alpha, -alpha, J[0], J[1], J[2]]
    
    # Define the list of corresponding observable terms
    obs = [
        qml.PauliZ(0), 
        qml.PauliZ(1), 
        qml.PauliX(0) @ qml.PauliX(1), 
        qml.PauliY(0) @ qml.PauliY(1), 
        qml.PauliZ(0) @ qml.PauliZ(1)
    ]
    
    # Use qml.dot to create the Hamiltonian from the coefficients and observables
    return qml.dot(coeffs, obs)




#4



n_bits = 2
dev =qml.device("default.qubit", wires = n_bits)

@qml.qnode(dev)
def two_close_spins(alpha, J, time, n):

    H = ham_close_spins(alpha, J)
    qml.TrotterProduct(-H, time, n)
    return qml.state()
