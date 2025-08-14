# https://pennylane.ai/codebook/pennylane-fundamentals/circuits-and-qnodes

#1.1a
def circuit(angle):
    """
    This quantum function implements the circuit shown above
    and returns the output quantum state
    """

    qml.Hadamard(wires = 0)
    qml.PauliX(wires = 1)
    qml.CNOT(wires = [0,1])
    qml.RY(angle, wires = 1)

    return qml.state()



#1.1b
dev_qubit = qml.device('default.qubit', wires = ["alice", "bob"])
dev_mixed = qml.device('default.mixed', wires = 2)

@qml.qnode(dev_qubit) # Choose the device you want
def example_circuit(theta):
    
    qml.RX(theta, wires = "alice" ) # Complete with wires in the device
    qml.CNOT(wires = ["alice", "bob"] ) # Complete with wires in the device
    
    return qml.state()




#1.1c


# Define the two-wire device using the "default.qubit" backend
dev = qml.device("default.qubit", wires=2)
circuit_qnode = qml.QNode(circuit, dev)
print(circuit_qnode(0.3))

#1.2a


def subcircuit_1(angle):
    qml.RX(angle, wires=0)
    qml.PauliY(wires=1)

def subcircuit_2():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def full_circuit(theta, phi):
    subcircuit_1(theta)
    subcircuit_2()
    subcircuit_1(phi)
    return qml.state()

print(full_circuit(0.3, 0.2))


#1.2b


def subcircuit_1(angle, wire_list):
    """
    Applies an RX(angle) on the first wire in wire_list
    and a PauliY on the second wire in wire_list.
    """
    qml.RX(angle, wires=wire_list[0])
    qml.PauliY(wires=wire_list[1])

def subcircuit_2(wire_list):
    """
    Applies a Hadamard on wire_list[0]
    and then a CNOT(control=wire_list[0], target=wire_list[1]).
    """
    qml.Hadamard(wires=wire_list[0])
    qml.CNOT(wires=wire_list) 

# Define the device
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def full_circuit(theta, phi):
    """
    Builds the full circuit by calling subcircuit_1 and subcircuit_2
    on different wire orderings.
    """
    # subcircuit_1 on wires [0, 1]
    subcircuit_1(theta, [0, 1])
    
    # subcircuit_2 on wires [0, 1]
    subcircuit_2([0, 1])
    
    # subcircuit_1 again, but now reversed wires [1, 0]
    subcircuit_1(phi, [1, 0])
    
    return qml.state()

print(full_circuit(0.3, 0.4))



#https://pennylane.ai/codebook/pennylane-fundamentals/quantum-operations

#2.1



dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def prep_circuit(alpha, beta, gamma):
    state = np.zeros(8, dtype=complex)
    state[1] = alpha
    state[2] = beta
    state[4] = gamma

    qml.StatePrep(state, wires=[0, 1, 2], normalize=True)


    return qml.state()

alpha = beta = gamma = 1/np.sqrt(3)
print("The prepared state is", prep_circuit(alpha, beta, gamma))


#2.2

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def single_qubit_gates(theta, phi):
    # Qubit 0
    qml.Hadamard(wires=0)
    qml.T(wires=0)
    qml.RX(theta, wires=0)

    # Qubit 1
    qml.Hadamard(wires=1)
    qml.S(wires=1)
    qml.RZ(phi, wires=1)
    
    return qml.state()

theta, phi = np.pi/3, np.pi/4
print("The output state of the circuit is: ", single_qubit_gates(theta, phi))



#2.3

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def multi_qubit_gates(theta, phi):
    qml.Hadamard(wires=0)
    qml.CRY(phi, wires=[0, 1])
    qml.CRX(theta, wires=[1, 2])
    qml.S(wires=1)
    qml.T(wires=2)
    qml.Toffoli(wires=[0, 1, 2])
    qml.SWAP(wires=[0, 2])

    return qml.state()

theta, phi = np.pi/3, np.pi/4
print("The output state is:\n", multi_qubit_gates(theta, phi))


#2.4


dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def ctrl_circuit(theta, phi):
    qml.RY(phi, wires=0)
    qml.Hadamard(wires=1)
    qml.RX(theta, wires=2)
    qml.ctrl(qml.S, control=0)(wires=1)
    qml.ctrl(qml.T, control=1, control_values=[0])(wires=2)
    qml.ctrl(qml.Hadamard, control=2)(wires=0)

    return qml.state()

theta, phi = np.pi/3, np.pi/4
print(ctrl_circuit(theta, phi))

#2.5


dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def phase_kickback(matrix):

    qml.Hadamard(wires=0)
    qml.ControlledQubitUnitary(matrix, control_wires=0, wires=1)
    qml.Hadamard(wires=0)

    return qml.state()


matrix = np.array([
    [-0.69165024-0.50339329j,  0.28335369-0.43350413j],
    [ 0.1525734 -0.4949106j , -0.82910055-0.2106588j ]
])

print("The state after phase kickback is:\n", phase_kickback(matrix))


#2.6


dev = qml.device("default.qubit", wires=3)

def do(k):
    qml.StatePrep([1, k], wires=[0], normalize=True)

def apply(theta):
    qml.IsingXX(theta, wires=[1, 2])

@qml.qnode(dev)
def do_apply_undo(k, theta):
    do(k)
    qml.ctrl(apply, control=0)(theta)
    qml.adjoint(do)(k)

    return qml.state()

k, theta = 0.5, 0.8
print("The output state is:\n", do_apply_undo(k, theta))




#https://pennylane.ai/codebook/pennylane-fundamentals/measurements-in-pennylane


#3.1

dev = qml.device("default.qubit", wires=2, shots=1)

@qml.qnode(dev)
def circuit():

    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

    return qml.sample(wires=[0, 1])

print(circuit())  


#3.2



dev = qml.device("default.qubit", wires=2)

# observable
A = np.array([[1, 0],
              [0, -1]])

@qml.qnode(dev)
def circuit():
    # prepare the Bell state
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

    # return the expectation value of A on qubit 0
    return qml.expval(qml.Hermitian(A, wires=0))

print("⟨A⟩ =", circuit())




#3.3


dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit():

    qml.H(wires=0)
    qml.CNOT(wires=[0,1])
    return qml.probs(op=qml.Z(wires=0)@qml.Z(wires=1))


#https://pennylane.ai/codebook/pennylane-fundamentals/circuit-optimization

#4.1


dev = qml.device("default.qubit", wires = 3)

@qml.qnode(dev)
def circuit_as_function(params):

    theta_0, theta_1, theta_2, theta_3 = params[0], params[1], params[2], params[3]
    
    qml.RX(theta_0, wires=0)
    qml.CNOT(wires=[0, 1])

    
    # To implement the control on |0> for wire 2, we frame the
    # Toffoli gate with PauliX gates on that wire.
    qml.PauliX(wires=2)
    qml.Toffoli(wires=[1, 2, 0])
    qml.PauliX(wires=2)
    
    # Apply the final rotation gates
    qml.RY(theta_1, wires=0)
    qml.RY(theta_2, wires=1)
    qml.RY(theta_3, wires=2)
    
    # Return the expectation value of PauliZ on the first wire
    return qml.expval(qml.PauliZ(0))

# The rest of your code remains the same
angles = np.linspace(0, 4 * np.pi, 200)
output_values = np.array([circuit_as_function([0.5, t, 0.5, 0.5]) for t in angles])




#4.2



dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def strong_entangler(weights):

    qml.StronglyEntanglingLayers(weights=weights, wires=range(4))
    
    # Return the expectation value of the PauliZ operator on the first qubit
    return qml.expval(qml.PauliZ(0))

# Define the shape for the weights. 
shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=4)

# Create some random weights with the correct shape
test_weights = np.random.uniform(0, 2 * np.pi, size=shape)
print("The shape of the weights is: ", test_weights.shape)




#4.3


dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def embedding_and_circuit(features, params):

    # Use the features to perform angle embedding on all wires
    qml.AngleEmbedding(features, wires=range(3))
    
    # Apply the sequence of CNOT gates as shown in the diagram
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    
    # Apply the RY rotation gates with the trainable parameters
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RY(params[2], wires=2)
    
    return qml.expval(qml.PauliZ(0))

# Define the input arrays
features = np.array([0.3, 0.4, 0.6], requires_grad=False)
params = np.array([0.4, 0.7, 0.9], requires_grad=True)

# Calculate and print the Jacobian
gradient = qml.jacobian(embedding_and_circuit)(features, params)
print("The gradient of the circuit is:", gradient)

# get the number of components from the gradient array
print("Number of components in the gradient:", len(gradient))


#4.4

dev = qml.device("default.qubit", wires = 2)

@qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
def circuit_for_hessian(params):

    qml.RY(params[0], wires=0)
    qml.IsingXX(params[1], wires=[0, 1])
    qml.RX(params[2], wires=0)
    qml.RX(params[3], wires=1)

    # Return the expectation value of the observable
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

test_params = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)

grad_fn = qml.grad(circuit_for_hessian, argnum=0)
hessian = qml.jacobian(grad_fn)(test_params)

print("The hessian of the circuit is: \n", hessian)



#4.5a
def cost_function(params):

  return circuit_as_function(params)**3-1/2*circuit_as_function(params)**2 + circuit_as_function(params)


#b


def optimize(cost_function, init_params, steps):

    opt = qml.GradientDescentOptimizer(stepsize = 0.4) 

    params = init_params

    for i in range(steps):

        params = opt.step(cost_function, params)

    return params, cost_function(params)

minimum = optimize(cost_function, np.array([0.1,0.2,0.3,0.4], requires_grad = True), 100)[1]



#https://pennylane.ai/codebook/pennylane-fundamentals/dynamic-circuits 

# (Dont work, example of internet)

#5.1
n_shots = 10000
dev = qml.device("default.qubit", shots=n_shots)
np.random.seed(0)


@qml.qnode(dev)
def circuit():
    """
    This quantum function implements the 'bomb tester' for a live bomb using mid-circuit measurements
    and returns relevant statistics with qml.counts
    """

    # 1. Primer divisor de haz
    qml.Hadamard(wires=0)

    # 2. Medida a mitad de circuito (la bomba)
    m_bomb = qml.measure(0)

    # 3. Segundo divisor de haz, aplicado incondicionalmente
    qml.Hadamard(wires=0)

    # 4. CORRECCIÓN: Si la bomba explotó (m_bomb=1), aplicamos otro Hadamard
    # para revertir el paso 3 y dejar el qubit en el estado de explosión |1>.
    qml.cond(m_bomb, qml.Hadamard)(wires=0)

    # 5. Medida final en los detectores
    m_det = qml.measure(0)

    # 6. Recolectar las estadísticas conjuntas para m_bomb y m_det
    return qml.counts(op=[m_bomb, m_det])

results = circuit()

# El caso favorable es '00': m_bomb=0 (sin explosión) y m_det=0 (detector D)
# El total de casos es el número de disparos (shots)
prob_suc = results.get('00', 0) / n_shots

print('the success probability is', prob_suc)


# (Dont work, example of internet)
#5.2

n_shots = 10000
dev = qml.device("default.qubit", shots=n_shots)
np.random.seed(0)

@qml.qnode(dev)
def circuit():
    """
    This quantum function implements an improved version of 'bomb tester'
    and returns relevant statistics with qml.counts
    """

    # --- Start of the previous solution ---
    # First beam-splitter
    qml.Hadamard(wires=0)

    # The bomb acts as a mid-circuit measurement
    m_bomb = qml.measure(0)

    # Second beam-splitter (unconditional)
    qml.Hadamard(wires=0)
    
    # Correction: If bomb exploded, apply H again to revert to |1>
    qml.cond(m_bomb, qml.Hadamard)(wires=0)
    
    # First set of detectors
    m_det = qml.measure(0)
    # --- End of the previous solution ---

    
    # --- Start of the new improvement logic (four lines) ---
    # Create a value that is 1 only for the uncertain case (m_bomb=0, m_det=1)
    m_uncertain_path = (1 - m_bomb) * m_det
    
    # If the photon is from the uncertain path, apply a rotation.
    # RY(pi/3) transforms |1> into a state with a 25% chance of being measured as 0.
    qml.cond(m_uncertain_path, qml.RY)(np.pi / 3, wires=0)
    
    # --- End of the new improvement logic ---

    # Measure the qubit after the potential recycling (second set of detectors)
    m_det_2 = qml.measure(0)
    
    # Return two sets of statistics as requested by the problem
    return qml.counts(op=[m_bomb, m_det]), qml.counts(op=[m_det, m_det_2])


results = circuit()  # array of two dictionaries

# Success from the first pass (m_bomb=0, m_det=0)
prob_suc_1 = results[0]["00"] / n_shots

# Success from the recycled pass (m_det=1, m_det_2=0)
prob_suc_2 = results[1]["10"] / n_shots

# Total success probability
prob_suc = prob_suc_1 + prob_suc_2
print("The success probability is", prob_suc)





#https://pennylane.ai/codebook/pennylane-fundamentals/inspecting-circuits
#6.1

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def circuit():
    """
    Implements a circuit and returns the state
    """
    qml.Hadamard(wires=0)
    qml.CRY(np.pi / 4, wires=(0, 1))
    qml.CRX(np.pi / 3, wires=(1, 2))
    qml.S(wires=1)
    qml.T(wires=2)
    qml.Toffoli(wires=(0, 1, 2))
    qml.SWAP(wires=(0, 2))
    return qml.state()


print(qml.draw(circuit)())  # use qml.draw()





#6.2

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def circuit():
    """
    Implements a circuit and returns the state
    """
    qml.Hadamard(wires=0)
    qml.CRY(np.pi / 4, wires=(0, 1))
    qml.CRX(np.pi / 3, wires=(1, 2))
    qml.Snapshot("very_important_state")
    qml.S(wires=1)
    qml.T(wires=2)
    qml.Toffoli(wires=(0, 1, 2))
    qml.Snapshot(measurement=qml.expval(qml.Z(0)))
    qml.SWAP(wires=(0, 2))
    return qml.state()


for key, val in qml.snapshots(circuit)().items(): print(key, val)
