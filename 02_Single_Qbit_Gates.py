#https://pennylane.ai/codebook/single-qubit-gates/x-and-h

#1

dev = qml.device("default.qubit", wires=1)

U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


@qml.qnode(dev)
def varied_initial_state(state):


    if state == 1:
        qml.PauliX(wires=0) 
    
    qml.QubitUnitary(U, wires=0)
    # KEEP THE QUBIT IN |0> OR CHANGE IT TO |1> DEPENDING ON THE state PARAMETER

    # APPLY U TO THE STATE

    return qml.state()


#2

dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def apply_hadamard():


    # APPLY THE HADAMARD GATE
    qml.Hadamard(wires=0)
    # RETURN THE STATE



  #3




  dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def apply_hadamard_to_state(state):


    # KEEP THE QUBIT IN |0> OR CHANGE IT TO |1> DEPENDING ON state
    if state == 1:
        qml.PauliX(wires=0) 
    # APPLY THE HADAMARD GATE
    qml.Hadamard(wires=0)
    # RETURN THE STATE
    return  qml.state()


print(apply_hadamard_to_state(0))
print(apply_hadamard_to_state(1))





#4




dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def apply_hxh(state):

    if (state):
       qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
 
    
    return  qml.state()    

# Print your results
print(apply_hxh(0))
print(apply_hxh(1))





#https://pennylane.ai/codebook/single-qubit-gates/its-just-a-phase



#1

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def apply_z_to_plus():
    qml.Hadamard(wires=0)
    qml.PauliZ(wires=0)
    return qml.state()

print(apply_z_to_plus())



#2


dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def fake_z():

    # Create the |+> state using Hadamard gate
    qml.Hadamard(wires=0)
    
    # Apply RZ with angle π to mimic the PauliZ operation
    qml.RZ(np.pi, wires=0)
    
    # Return the final state
    return qml.state()

print(fake_z())

#3


dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def many_rotations():
    """Implement the circuit depicted above and return the quantum state.

    Returns:
        np.array[complex]: The state of the qubit after the operations.
    """

    # Apply the gates in the order shown in the circuit
    qml.Hadamard(wires=0)           # H
    qml.S(wires=0)                  # S
    qml.adjoint(qml.T)(wires=0)     # T
    qml.RZ(0.3, wires=0)            # Rz(0.3)
    qml.adjoint(qml.S)(wires=0)     # S
    
    # Return the final state
    return qml.state()



  #https://pennylane.ai/codebook/single-qubit-gates/from-a-different-angle




  #1


  dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def apply_rx_pi(state):
    if state == 1:
        qml.PauliX(wires=0)

    qml.RX(np.pi, wires=0)  

    return qml.state() 

print(apply_rx_pi(0)) 
print(apply_rx_pi(1))  



#2


dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def apply_rx(theta, state):

    if state == 1:
        qml.PauliX(wires=0)

    qml.RX(theta, wires=0) 

    return qml.state()  


# Code for plotting
angles = np.linspace(0, 4 * np.pi, 200)
output_states = np.array([apply_rx(t, 0) for t in angles])

plot = plotter(angles, output_states)



#3



dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def apply_ry(theta, state):

    if state == 1:
        qml.PauliX(wires=0)

    qml.RY(theta, wires=0) 

    return qml.state()  

# Code for plotting
angles = np.linspace(0, 4 * np.pi, 200)
output_states = np.array([apply_ry(t, 0) for t in angles])

plot = plotter(angles, output_states)




#https://pennylane.ai/codebook/single-qubit-gates/universal-gate-sets


#1

dev = qml.device("default.qubit", wires=1)



phi, theta, omega = np.pi/2, np.pi/2, np.pi/2

@qml.qnode(dev)
def hadamard_with_rz_rx(phi, theta, omega):
    qml.RX(theta, wires=0)
    qml.RZ(omega, wires=0)
    return qml.state()

hadamard_with_rz_rx(phi, theta, omega)


#2

dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def convert_to_rz_rx():
   
    qml.RX(np.pi/2, wires=0)       # From H
    qml.RZ(7*np.pi/4, wires=0)      # Combined RZs: π + π/2 + π/4
    qml.RX(np.pi, wires=0)         # From Y


    return qml.state()

convert_to_rz_rx()


#3


dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def unitary_with_h_and_t():
    qml.H(wires=0)
    qml.T(wires=0)
    qml.H(wires=0)
    qml.T(wires=0)
    qml.T(wires=0)
    qml.H(wires=0)
    return qml.state()




  #https://pennylane.ai/codebook/single-qubit-gates/prepare-yourself

  #1

  dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def prepare_state():

    qml.H(wires=0)
    qml.RZ(5 * np.pi / 4, wires=0)

    return qml.state()




  #2


  dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def prepare_state():
    #  cos(θ/2)=√3/2, sin(θ/2)=-1/2
    qml.RX(np.pi/3, wires=0)


    return qml.state()




  #3


  v = np.array([0.52889389 - 0.14956775j, 0.67262317 + 0.49545818j])

dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def prepare_state(state=v):

    qml.StatePrep(v, wires=0)
    return qml.state()


# This will draw the quantum circuit 
print(prepare_state(v))
print()
print(qml.draw(prepare_state, level="device")(v))





#https://pennylane.ai/codebook/single-qubit-gates/measurements



#1


dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def apply_h_and_measure(state):

    if state == 1:
        qml.PauliX(wires=0)


    qml.H(wires=0)
    return qml.probs(wires=0)


print(apply_h_and_measure(0))
print(apply_h_and_measure(1))



#2


# WRITE A QUANTUM FUNCTION THAT PREPARES (1/2)|0> + i(sqrt(3)/2)|1>
def prepare_psi():
    qml.RX(-2 * np.pi / 3, wires=0)


# WRITE A QUANTUM FUNCTION THAT SENDS BOTH |0> TO |y_+> and |1> TO |y_->
def y_basis_rotation():
    qml.Hadamard(0)
    qml.S(0)



  #3


  dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def measure_in_y_basis():

    prepare_psi()
    #(Inverse function)
    qml.adjoint(y_basis_rotation)()
    return qml.probs(wires=0)

print(measure_in_y_basis())



#https://pennylane.ai/codebook/single-qubit-gates/what-did-you-expect



#1


dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit():
    qml.RX(np.pi/4, wires=0)
    qml.Hadamard(wires=0)
    qml.PauliZ(wires=0)
    return qml.expval(qml.PauliY(0)) 

print(circuit())

#2


# An array to store your results
shot_results = []

# Different numbers of shots
shot_values = [100, 1000, 10000, 100000, 1000000]

for shots in shot_values:
    
    dev = qml.device("default.qubit", wires=1, shots=shots)

    @qml.qnode(dev)
    def circuit_y(shots=shots): 
      
        qml.RX(np.pi/4, wires=0)
        qml.Hadamard(wires=0)
        qml.PauliZ(wires=0)
        return qml.expval(qml.PauliY(0))

    shot_results.append(circuit_y())

print(qml.math.unwrap(shot_results))



#3


dev = qml.device("default.qubit", wires=1, shots=100000)


@qml.qnode(dev)
def circuit():
    qml.RX(np.pi / 4, wires=0)
    qml.Hadamard(wires=0)
    qml.PauliZ(wires=0)
    return qml.sample(qml.PauliY(wires=0))

    
def compute_expval_from_samples(samples):
    estimated_expval = np.mean(samples)
    return estimated_expval

    return estimated_expval


samples = circuit()
print(compute_expval_from_samples(samples))



#4


def variance_experiment(n_shots):

    n_trials = 100


    dev = qml.device("default.qubit", wires=1, shots=n_shots)


    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    results = [circuit() for _ in range(n_trials)]
    return np.var(results)


def variance_scaling(n_shots):
    return 1.0 / n_shots



shot_vals = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000]
results_experiment = [variance_experiment(s) for s in shot_vals]
results_scaling   = [variance_scaling(s)   for s in shot_vals]


plot = plotter(shot_vals, results_experiment, results_scaling)




  

    return  qml.state()
