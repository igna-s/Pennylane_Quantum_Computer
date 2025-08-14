
#https://pennylane.ai/codebook/introduction-to-quantum-computing/all-about-qubits

#1.1

ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])

def normalize_state(alpha, beta):

    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    
    return np.array([alpha / norm, beta / norm])

#1.2


def inner_product(state_1, state_2):
    return np.vdot(state_1, state_2)

# Test your results with this code
ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])

print(f"<0|0> = {inner_product(ket_0, ket_0)}")
print(f"<0|1> = {inner_product(ket_0, ket_1)}")
print(f"<1|0> = {inner_product(ket_1, ket_0)}")
print(f"<1|1> = {inner_product(ket_1, ket_1)}")



#1.3



def measure_state(state, num_meas):

    prob_0 = np.abs(state[0])**2
    prob_1 = np.abs(state[1])**2

    outcomes = [0, 1]
    probabilities = [prob_0, prob_1]

    measurements = np.random.choice(outcomes, size=num_meas, p=probabilities)

    return measurements



#1.4


U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def apply_u(state):
    return U @ state


#1.5


U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def initialize_state():
    """Prepare a qubit in state |0>."""
    return np.array([1, 0])  # Estado |0>

def apply_u(state):
    """Apply a quantum operation."""
    return np.dot(U, state)  # Aplica la puerta Hadamard al estado

def measure_state(state, num_meas):
    """Measure a quantum state num_meas times."""
    p_alpha = np.abs(state[0]) ** 2
    p_beta = np.abs(state[1]) ** 2
    meas_outcome = np.random.choice([0, 1], p=[p_alpha, p_beta], size=num_meas)
    return meas_outcome

def quantum_algorithm():

    state = initialize_state()  # Inicializa el estado |0⟩
    state = apply_u(state)      # Aplica la puerta Hadamard
    return measure_state(state, 100)  # Realiza 100 mediciones




#https://pennylane.ai/codebook/introduction-to-quantum-computing/quantum-circuits



#1.1

def my_circuit(theta, phi):


    # REORDER THESE 5 GATES TO MATCH THE CIRCUIT IN THE PICTURE

    qml.CNOT(wires=[0, 1])
    qml.RX(theta, wires=2)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[2, 0])
    qml.RY(phi, wires=1)

    # This is the measurement; we return the probabilities of all possible output states
    # You'll learn more about what types of measurements are available in a later node
    return qml.probs(wires=[0, 1, 2])



#1.2


# This creates a device with three wires on which PennyLane can run computations
dev = qml.device("default.qubit", wires=3)


def my_circuit(theta, phi, omega):


    # Here are two examples, so you can see the format:
    # qml.CNOT(wires=[0, 1])
    # qml.RX(theta, wires=0)

    qml.RX(theta, wires=0)
    qml.RY(phi, wires=1)
    qml.RZ(omega, wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    
    return qml.probs(wires=[0, 1, 2])


# This creates a QNode, binding the function and device
my_qnode = qml.QNode(my_circuit, dev)

# We set up some values for the input parameters
theta, phi, omega = 0.1, 0.2, 0.3

# Now we can execute the QNode by calling it like we would a regular function
my_qnode(theta, phi, omega)


#1.3


dev = qml.device("default.qubit", wires=3)

# DECORATE THE FUNCTION BELOW TO TURN IT INTO A QNODE

@qml.qnode(dev)
def my_circuit(theta, phi, omega):
    qml.RX(theta, wires=0)
    qml.RY(phi, wires=1)
    qml.RZ(omega, wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    return qml.probs(wires=[0, 1, 2])


theta, phi, omega = 0.1, 0.2, 0.3


my_circuit(theta,phi,omega)
# RUN THE QNODE WITH THE PROVIDED PARAMETERS



#1.4


dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def my_circuit(theta, phi, omega):
    qml.RX(theta, wires=0)
    qml.RY(phi, wires=1)
    qml.RZ(omega, wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    return qml.probs(wires=[0, 1, 2])



# FILL IN THE CORRECT CIRCUIT DEPTH
depth = 4




#https://pennylane.ai/codebook/introduction-to-quantum-computing/unitary-matrices


#1

dev = qml.device("default.qubit", wires=1)

U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


@qml.qnode(dev)
def apply_u():

    # USE QubitUnitary TO APPLY U TO THE QUBIT
    qml.QubitUnitary(U, wires=0)
    # Return the state
    return qml.state()


#2


dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def apply_u_as_rot(phi, theta, omega):

    # APPLY A ROT GATE USING THE PROVIDED INPUT PARAMETERS
    qml.Rot(phi, theta, omega, wires=0)
    # RETURN THE QUANTUM STATE VECTOR

    return qml.state()
