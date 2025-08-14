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
    qml.CNOT(wires=wire_list)  # [control, target]

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

# Example usage
print(full_circuit(0.3, 0.4))


