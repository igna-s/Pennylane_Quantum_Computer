#https://pennylane.ai/codebook/circuits-with-many-qubits/multi-qubit-systems

#1

num_wires = 3
dev = qml.device("default.qubit", wires=num_wires)


@qml.qnode(dev)
def make_basis_state(basis_id):

    # CREATE THE BASIS STATE
    bits = [int(x) for x in np.binary_repr(basis_id, width=num_wires)]

    for i, bit in enumerate(bits):
        if bit == 1:
            qml.PauliX(wires=i)

    return qml.state()


basis_id = 3
print(f"Output state = {make_basis_state(basis_id)}")


#2

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def two_qubit_circuit():

    # PREPARE |+>|1>
    qml.Hadamard(wires=0)
    qml.PauliX(wires=1)

    # RETURN TWO EXPECTATION VALUES, Y ON FIRST QUBIT, Z ON SECOND QUBIT
    return qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(1))


results = two_qubit_circuit()



#3


dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def create_one_minus():
    # PREPARE |1>|->

    qml.PauliX(wires=0)
    qml.PauliX(wires=1)
    qml.Hadamard(wires=1)

    # RETURN A SINGLE EXPECTATION VALUE Z \otimes X
    return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))


print(create_one_minus())





#4


dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def circuit_1(theta):

    # Apply the gates as described in the circuit diagram
    qml.RX(theta, wires=0)
    qml.RY(2 * theta, wires=1)
    
    # Return the expectation value of Z on each qubit
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))


@qml.qnode(dev)
def circuit_2(theta):

    # Apply the gates as described in the circuit diagram
    qml.RX(theta, wires=0)
    qml.RY(2 * theta, wires=1)

    # Return the expectation value of the Z_0 Z_1 observable
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


def zi_iz_combination(ZI_results, IZ_results):
    # For separable states, the expectation value of the tensor product
    # is the product of the individual expectation values.
    combined_results = ZI_results * IZ_results
    return combined_results


theta = np.linspace(0, 2 * np.pi, 100)

# Run circuit 1, and process the results
circuit_1_results = np.array([circuit_1(t) for t in theta])

ZI_results = circuit_1_results[:, 0]
IZ_results = circuit_1_results[:, 1]
combined_results = zi_iz_combination(ZI_results, IZ_results)

# Run circuit 2
ZZ_results = np.array([circuit_2(t) for t in theta])

# Plot your results
plot = plotter(theta, ZI_results, IZ_results, ZZ_results, combined_results)

print(f"Expectation values [Y(0), Z(1)]: {results}")




#https://pennylane.ai/codebook/circuits-with-many-qubits/all-tied-up



#1


num_wires = 2
dev = qml.device("default.qubit", wires=num_wires)


@qml.qnode(dev)
def apply_cnot(basis_id):

    # Prepare the basis state 
    bits = [int(x) for x in np.binary_repr(basis_id, width=num_wires)]
    qml.BasisState(bits, wires=[0, 1])

    # APPLY THE CNOT
    qml.CNOT(wires=[0, 1])

    return qml.state()

# REPLACE THE BIT STRINGS VALUES BELOW WITH THE CORRECT ONES
cnot_truth_table = {"00": "00", "01": "01", "10": "11", "11": "10"}




#2


dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def apply_h_cnot():
    # Start in |00>
    qml.Hadamard(wires=0)      # H on qubit 0
    qml.CNOT(wires=[0, 1])     # CNOT: control=0, target=1
    return qml.state()

# 3) Run it and inspect the statevector
state = apply_h_cnot()
print("Output statevector:", state)

# 4) Classify separability
state_status = "entangled"
print("State is:", state_status)


#3

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def controlled_rotations(theta, phi, omega):
    # 1) initial |000⟩
    qml.Hadamard(wires=0)
    
    qml.CRX(theta,wires=[0, 1])
    qml.CRY(phi, wires=[1, 2])
    qml.CRZ(omega, wires=[0, 1])

    return qml.probs(wires=[0, 1, 2])


theta, phi, omega = 0.1, 0.2, 0.3
probs = controlled_rotations(theta, phi, omega)
print("Outcome probabilities:", probs)





#https://pennylane.ai/codebook/circuits-with-many-qubits/weve-got-it-under-control


#1
dev = qml.device("default.qubit", wires=2)

# Prepare a two-qubit state; change up the angles if you like
phi, theta, omega = 1.2, 2.3, 3.4


@qml.qnode(device=dev)
def true_cz(phi, theta, omega):
    prepare_states(phi, theta, omega)


    # IMPLEMENT THE REGULAR CZ GATE HERE
    qml.CZ(wires=[0, 1])

    return qml.state()


@qml.qnode(dev)
def imposter_cz(phi, theta, omega):
    prepare_states(phi, theta, omega)


    # IMPLEMENT CZ USING ONLY H AND CNOT
    qml.H(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.H(wires=1)

    return qml.state()


print(f"True CZ output state {true_cz(phi, theta, omega)}")
print(f"Imposter CZ output state {imposter_cz(phi, theta, omega)}")




#2


dev = qml.device("default.qubit", wires=2)

# Prepare a two-qubit state; change up the angles if you like
phi, theta, omega = 1.2, 2.3, 3.4


@qml.qnode(dev)
def apply_swap(phi, theta, omega):
    prepare_states(phi, theta, omega)



    # IMPLEMENT THE REGULAR SWAP GATE HERE
    qml.SWAP(wires=[0, 1])

    return qml.state()


@qml.qnode(dev)
def apply_swap_with_cnots(phi, theta, omega):
    prepare_states(phi, theta, omega)


    # IMPLEMENT THE SWAP GATE USING A SEQUENCE OF CNOTS
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[0, 1])
    
    return qml.state()


print(f"Regular SWAP state = {apply_swap(phi, theta, omega)}")
print(f"CNOT SWAP state = {apply_swap_with_cnots(phi, theta, omega)}")



#3


dev = qml.device("default.qubit", wires=3)

# Prepare first qubit in |1>, and arbitrary states on the second two qubits
phi, theta, omega = 1.2, 2.3, 3.4


# A helper function just so you can visualize the initial state
# before the controlled SWAP occurs.
@qml.qnode(dev)
def no_swap(phi, theta, omega):
    prepare_states(phi, theta, omega)
    return qml.state()


@qml.qnode(dev)
def controlled_swap(phi, theta, omega):
    prepare_states(phi, theta, omega)



    qml.Toffoli(wires=[0, 1, 2])
    qml.Toffoli(wires=[0, 2, 1])
    qml.Toffoli(wires=[0, 1, 2])

    return qml.state()


print(no_swap(phi, theta, omega))
print(controlled_swap(phi, theta, omega))




#4

dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev)
def four_qubit_mcx():
    for i in range(3):
        qml.Hadamard(wires=i)

    # IMPLEMENT THE CIRCUIT ABOVE USING A 4-QUBIT MULTI-CONTROLLED X
    qml.MultiControlledX(wires=[0, 1, 2, 3], control_values=[0,0,1])
    return qml.state()


print(four_qubit_mcx())


#5

# Wires 0, 1, 2 are the control qubits
# Wire 3 is the auxiliary qubit
# Wire 4 is the target
dev = qml.device("default.qubit", wires=5)


@qml.qnode(dev)
def four_qubit_mcx_only_tofs():
    # We will initialize the control qubits in state |1> so you can see
    # how the output state gets changed.
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)
    qml.PauliX(wires=2)


    # IMPLEMENT A 3-CONTROLLED NOT WITH TOFFOLIS
    qml.Toffoli(wires=[0, 1, 3])
    qml.Toffoli(wires=[2, 3, 4])
    qml.Toffoli(wires=[0, 1, 3])

    return qml.state()


# print(four_qubit_mcx_only_tofs())



#https://pennylane.ai/codebook/circuits-with-many-qubits/multi-qubit-challenge


#1


dev = qml.device("default.qubit", wires=2)

# Starting from the state |00>, implement a PennyLane circuit
# to construct each of the Bell basis states.


@qml.qnode(dev)
def prepare_psi_plus():

    # PREPARA (1/sqrt(2)) (|00> + |11>)
    # Initial State: |00>
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

    return qml.state()


@qml.qnode(dev)
def prepare_psi_minus():

    # Prepares (1/sqrt(2)) (|00> - |11>)
    # Initial State: |10>
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

    return qml.state()


@qml.qnode(dev)
def prepare_phi_plus():

    # Prepares (1/sqrt(2)) (|01> + |10>)
    # Initial State: |01>
    qml.PauliX(wires=1)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

    return qml.state()


@qml.qnode(dev)
def prepare_phi_minus():

    # Prepares (1/sqrt(2)) (|01> - |10>)
    # Initial State: |11>
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

    return qml.state()


psi_plus = prepare_psi_plus()
psi_minus = prepare_psi_minus()
phi_plus = prepare_phi_plus()
phi_minus = prepare_phi_minus()

# Uncomment to print results
# print(f"|ψ_+> = {psi_plus}")
# print(f"|ψ_-> = {psi_minus}")
# print(f"|ϕ_+> = {phi_plus}")
# print(f"|ϕ_-> = {phi_minus}")



#2

dev = qml.device("default.qubit", wires=3)

# State of first 2 qubits
state = [0, 1]


@qml.qnode(dev)
def apply_control_sequence(state):
    # Set up initial state of the first two qubits
    if state[0] == 1:
        qml.PauliX(wires=0)
    if state[1] == 1:
        qml.PauliX(wires=1)

    # Set up initial state of the third qubit - use |->
    # so we can see the effect on the output
    qml.PauliX(wires=2)
    qml.Hadamard(wires=2)


    # IMPLEMENT THE MULTIPLEXER
    # IF STATE OF FIRST TWO QUBITS IS 01, APPLY X TO THIRD QUBIT
    qml.PauliX(wires=0)
    qml.Toffoli(wires=[0, 1, 2])
    qml.PauliX(wires=0)

    # IF STATE OF FIRST TWO QUBITS IS 10, APPLY Z TO THIRD QUBIT
    qml.PauliX(wires=1)
    qml.Hadamard(wires=2)
    qml.Toffoli(wires=[0, 1, 2])
    qml.PauliX(wires=1)
    qml.Hadamard(wires=2)

    # IF STATE OF FIRST TWO QUBITS IS 11, APPLY Y TO THIRD QUBIT
    qml.adjoint(qml.S)(wires=2)
    qml.Toffoli(wires=[0, 1, 2])
    qml.S(wires=2)

    return qml.state()







